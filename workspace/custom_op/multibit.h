// multibit.h
#ifndef MULTIBIT_H_
#define MULTIBIT_H_

template <typename Device, typename T>
struct MultibitFunctor {
  void operator()(const Device& d, int size, const int *b, const T* in, T* out);
};

#endif
