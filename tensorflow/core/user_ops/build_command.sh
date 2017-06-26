bazel build -c opt --config=cuda //tensorflow/workspace/custom_op:binarize.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
bazel build -c opt --config=cuda //tensorflow/workspace/custom_op:multibit.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
