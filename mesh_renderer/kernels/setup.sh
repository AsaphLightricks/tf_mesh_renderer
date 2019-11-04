#!/bin/bash


#TF_INC=/Users/ashamir/PycharmProjects/syntheticMotionBlur/venv/lib/python3.7/site-packages/tensorflow_core/include
#TF_LIB=/Users/ashamir/PycharmProjects/syntheticMotionBlur/venv/lib/python3.7/site-packages/tensorflow_core

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
SRCS='rasterize_triangles_grad.cc rasterize_triangles_op.cc rasterize_triangles_impl.cc'

#echo g++ -std=c++11 -shared $SRCS -o rasterize_triangles_kernel.so -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I $TF_INC -I $TF_INC/external/nsync/public -L $TF_LIB -ltensorflow_framework -O2
g++ -std=c++11 -shared $SRCS -o rasterize_triangles_kernel.so -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I $TF_INC -I $TF_INC/external/nsync/public -L $TF_LIB -ltensorflow_framework -O2
