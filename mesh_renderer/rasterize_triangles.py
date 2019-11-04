# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiable triangle rasterizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mesh_renderer import camera_utils

rasterize_triangles_module = tf.load_op_library('mesh_renderer/kernels/rasterize_triangles_kernel.so')


# os.path.join(os.environ['TEST_SRCDIR'],
# 'tf_mesh_renderer/mesh_renderer/kernels/rasterize_triangles_kernel.so'))


def rasterize(world_space_vertices, attributes, triangles, camera_matrices,
              image_width, image_height, background_value):
    """Rasterizes a mesh and computes interpolated vertex attributes.

  Applies projection matrices and then calls rasterize_clip_space().

  Args:
    world_space_vertices: 3-D float32 tensor of xyz positions with shape
      [batch_size, vertex_count, 3].
    attributes: 3-D float32 tensor with shape [batch_size, vertex_count,
      attribute_count]. Each vertex attribute is interpolated across the
      triangle using barycentric interpolation.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
      should contain vertex indices describing a triangle such that the
      triangle's normal points toward the viewer if the forward order of the
      triplet defines a clockwise winding of the vertices. Gradients with
      respect to this tensor are not available.
    camera_matrices: 3-D float tensor with shape [batch_size, 4, 4] containing
      model-view-perspective projection matrices.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.
    background_value: a 1-D float32 tensor with shape [attribute_count]. Pixels
      that lie outside all triangles take this value.

  Returns:
    A 4-D float32 tensor with shape [batch_size, image_height, image_width,
    attribute_count], containing the interpolated vertex attributes at
    each pixel.

  Raises:
    ValueError: An invalid argument to the method is detected.
  """
    clip_space_vertices = camera_utils.transform_homogeneous(
        camera_matrices, world_space_vertices)
    return rasterize_clip_space(clip_space_vertices, attributes, triangles,
                                image_width, image_height, background_value)


def rasterize_clip_space(clip_space_vertices, attributes, triangles,
                         image_width, image_height, background_value):
    """Rasterizes the input mesh expressed in clip-space (xyzw) coordinates.

  Interpolates vertex attributes using perspective-correct interpolation and
  clips triangles that lie outside the viewing frustum.

  Args:
    clip_space_vertices: 3-D float32 tensor of homogenous vertices (xyzw) with
      shape [batch_size, vertex_count, 4].
    attributes: 3-D float32 tensor with shape [batch_size, vertex_count,
      attribute_count]. Each vertex attribute is interpolated across the
      triangle using barycentric interpolation.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
      should contain vertex indices describing a triangle such that the
      triangle's normal points toward the viewer if the forward order of the
      triplet defines a clockwise winding of the vertices. Gradients with
      respect to this tensor are not available.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.
    background_value: a 1-D float32 tensor with shape [attribute_count]. Pixels
      that lie outside all triangles take this value.

  Returns:
    A 4-D float32 tensor with shape [batch_size, image_height, image_width,
    attribute_count], containing the interpolated vertex attributes at
    each pixel.

  Raises:
    ValueError: An invalid argument to the method is detected.
  """
    if not image_width > 0:
        raise ValueError('Image width must be > 0.')
    if not image_height > 0:
        raise ValueError('Image height must be > 0.')
    if len(clip_space_vertices.shape) != 3:
        raise ValueError('The vertex buffer must be 3D.')

    vertex_count = clip_space_vertices.shape[1].value

    batch_size = tf.shape(clip_space_vertices)[0]

    per_image_barycentric_coordinates = tf.TensorArray(dtype=tf.float32,
                                                       size=batch_size)
    per_image_vertex_ids = tf.TensorArray(dtype=tf.int32, size=batch_size)

    def batch_loop_condition(b, *args):
        return b < batch_size

    def batch_loop_iteration(b, per_image_barycentric_coordinates,
                             per_image_vertex_ids):
        barycentric_coords, triangle_ids, _ = (
            rasterize_triangles_module.rasterize_triangles(
                clip_space_vertices[b, :, :], triangles, image_width,
                image_height))
        per_image_barycentric_coordinates = \
            per_image_barycentric_coordinates.write(
                b, tf.reshape(barycentric_coords, [-1, 3]))

        vertex_ids = tf.gather(triangles, tf.reshape(triangle_ids, [-1]))
        reindexed_ids = tf.add(vertex_ids, b * clip_space_vertices.shape[1].value)
        per_image_vertex_ids = per_image_vertex_ids.write(b, reindexed_ids)

        return b + 1, per_image_barycentric_coordinates, per_image_vertex_ids

    b = tf.constant(0)
    _, per_image_barycentric_coordinates, per_image_vertex_ids = tf.while_loop(
        batch_loop_condition, batch_loop_iteration,
        [b, per_image_barycentric_coordinates, per_image_vertex_ids])

    barycentric_coordinates = tf.reshape(
        per_image_barycentric_coordinates.stack(), [-1, 3])
    vertex_ids = tf.reshape(per_image_vertex_ids.stack(), [-1, 3])

    # Indexes with each pixel's clip-space triangle's extrema (the pixel's
    # 'corner points') ids to get the relevant properties for deferred shading.
    flattened_vertex_attributes = tf.reshape(attributes,
                                             [batch_size * vertex_count, -1])
    corner_attributes = tf.gather(flattened_vertex_attributes, vertex_ids)

    # Computes the pixel attributes by interpolating the known attributes at the
    # corner points of the triangle interpolated with the barycentric coordinates.
    weighted_vertex_attributes = tf.multiply(
        corner_attributes, tf.expand_dims(barycentric_coordinates, axis=2))
    summed_attributes = tf.reduce_sum(weighted_vertex_attributes, axis=1)
    attribute_images = tf.reshape(summed_attributes,
                                  [batch_size, image_height, image_width, -1])

    # Barycentric coordinates should approximately sum to one where there is
    # rendered geometry, but be exactly zero where there is not.
    alphas = tf.clip_by_value(
        tf.reduce_sum(2.0 * barycentric_coordinates, axis=1), 0.0, 1.0)
    alphas = tf.reshape(alphas, [batch_size, image_height, image_width, 1])

    attributes_with_background = (
            alphas * attribute_images + (1.0 - alphas) * background_value)

    return attributes_with_background


# @tf.RegisterGradient('RasterizeTriangles')
# def _rasterize_triangles_grad(op, df_dbarys, df_dids, df_dz):
#   # Gradients are only supported for barycentric coordinates. Gradients for the
#   # z-buffer are not currently implemented. If you need gradients w.r.t. z,
#   # include z as a vertex attribute when calling rasterize_triangles.
#   del df_dids, df_dz
#   return rasterize_triangles_module.rasterize_triangles_grad(
#       op.inputs[0], op.inputs[1], op.outputs[0], op.outputs[1], df_dbarys,
#       op.get_attr('image_width'), op.get_attr('image_height')), None


if __name__ == '__main__':
    im_size = 500

    m1 = np.load('../meshes/m1.npz')
    m2 = np.load('../meshes/m2.npz')

    bbox = m2['bbox']
    orig = m2['orig']

    v1 = m1['pos'].astype(np.float32)
    v2 = m2['pos'].astype(np.float32)

    # v1[:, 0] *= -1
    # v2[:, 0] *= -1

    v1[:, 2] += -1200
    v2[:, 2] += -1200

    v1 = np.expand_dims(v1, axis=0)
    v2 = np.expand_dims(v2, axis=0)

    c1 = m1['colors'].astype(np.float32)
    c2 = m2['colors'].astype(np.float32)

    c1 = np.hstack([c1, np.ones((c1.shape[0], 1), dtype=c1.dtype)])
    c2 = np.hstack([c2, np.ones((c2.shape[0], 1), dtype=c2.dtype)])

    c1 = np.expand_dims(c1, axis=0)
    c2 = np.expand_dims(c2, axis=0)

    print('c1.mean():', c1.mean())

    fl1 = m1['fl']
    fl2 = m2['fl']

    triangles = np.load('/Users/ashamir/PycharmProjects/research-face-model/auxiliary_files/triangles.npy')
    triangles = triangles.astype(np.int32)

    print('triangles.min():', triangles.min())

    vertices = np.array([[[0.5, 0.5, -0.75],
                          [0.5, -1., -0.75],
                          [0., 0.5, -1],
                          [-1, 0, -1],
                          [-0.5, -0.5, -0.5],
                          [0, 0, -0.5],
                          [0.75, -0.5, -1]]], dtype=np.float32)

    colors = np.array([[[1., 1., 1.],
                        [0., 1., 0.],
                        [1., 1., 0.],
                        [1, 0, 0],
                        [1, 0, 1],
                        [0, 0, 1],
                        [0, 1, 1]]], dtype=np.float32)

    # triangles = np.array([[0, 1, 2],
    #                       [1, 2, 3],
    #                       [4, 5, 6]], dtype=np.int32)

    l = -0.25
    r = 0.25
    b = -0.25
    t = 0.25
    n = 0.1
    f = 2

    frustrum = np.array([[[2 * n / (r - l), 0, (r + l) / (r - l), 0],
                          [0, 2 * n / (t - b), (t + b) / (t - b), 0],
                          [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                          [0, 0, -1, 0]]], dtype=np.float32)

    fl = (1 + fl2) * 1200 / 120
    # fl = 10

    f_mat = np.array([[[fl, 0,  0, 0],
                       [0, fl,  0, 0],
                       [0,  0,  1, 0],
                       [0,  0, -1, 0]]], dtype=np.float32)

    eye = np.array([[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]], dtype=np.float32)

    background_val = np.array([[0, 0, 0, 0]], dtype=np.float32)

    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    out = rasterize(v2, c2, triangles, f_mat, image_width=w, image_height=h,
                    background_value=background_val)

    sess = tf.InteractiveSession()

    # frustrum = frustrum.squeeze()
    # vertices = vertices.squeeze()
    # verts = np.hstack([vertices, np.ones((vertices.shape[0], 1), dtype=vertices.dtype)])
    # proj = (frustrum @ verts.T).T



    out = np.flipud(out.eval().squeeze())
    rgb = out[..., :-1]
    mask = out[..., -1]
    # mask = binary_erosion(mask, disk(1))

    orig = orig / 255
    j, i = bbox[0], bbox[1]
    orig[i: i + h, j: j + w] = orig[i: i + h, j: j + w] * (1 - mask[..., None]) + rgb * mask[..., None]

    # np.set_printoptions(precision=2)
    # print(proj)
    # print(proj[:, :3] / proj[:, 3, None])

    # print(im.shape)
    # print(im)
    # f, ax = plt.subplots(1, 4)
    # ax[2].imshow(rgb)
    # ax[3].imshow(mask, cmap='gray')
    # ax[0].imshow(out)
    # ax[1].imshow(orig)
    # plt.show()
    import imageio
    imageio.imsave('daniel.png', orig)
