load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

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

tf_custom_op_library(
    name = "multibit2.so",
    srcs = ["multibit2.cc", "multibit2.h"],
    gpu_srcs = ["multibit2.cu.cc", "multibit2.h"],
)
