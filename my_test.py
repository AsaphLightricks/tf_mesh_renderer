
import tensorflow as tf

rasterize_triangles_module = tf.load_op_library('rasterize_triangles_kernel.so')

print(dir(rasterize_triangles_module))
print(rasterize_triangles_module.__doc__)