#ifndef SOFT_MAX
#define SOFT_MAX

#include "tensor.h"

// преобразует значения вектора в вероятности
void soft_max(float *vector);

// поиск градиента
void soft_max_gradient(float *batch, Tensor *tensor);

#endif