#include "tensor.h"


// преобразует тензор в матрицу для свертки
void im2col(Tensor *tensor, int size_ker, float *res);