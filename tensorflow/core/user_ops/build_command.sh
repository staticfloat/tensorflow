bazel build -c opt --config=cuda //tensorflow/core/user_ops:binarize.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
bazel build -c opt --config=cuda //tensorflow/core/user_ops:multibit.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
