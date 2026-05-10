#ifndef IM2COL_H
#define IM2COL_H


#include "tensor.h"


// преобразует тензор в матрицу для свертки
float *im2col(Tensor *tensor, int size_ker, int *res_row, int *res_col);
// tensor - входное изображение
// size_ker - размер ядра
// res_row * res_col - размер полученной матрицы

#endif