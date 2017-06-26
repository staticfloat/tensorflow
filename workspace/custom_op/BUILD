load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "cuda_op_kernel.so",
    srcs = ["cuda_op_kernel.cc"],
    gpu_srcs = ["cuda_op_kernel.cu.cc"],
)

tf_custom_op_library(
    name = "example.so",
    srcs = ["example.cc", "example.h"],
    gpu_srcs = ["example.cu.cc", "example.h"],
)

tf_custom_op_library(
    name = "binarize.so",
    srcs = ["binarize.cc", "binarize.h"],
    gpu_srcs = ["binarize.cu.cc", "binarize.h"],
)

tf_custom_op_library(
    name = "multibit.so",
    srcs = ["multibit.cc", "multibit.h"],
    gpu_srcs = ["multibit.cu.cc", "multibit.h"],
)
