#ifndef COL2IM
#define COL2IM

#include "tensor.h"

// превращает матрицу обратно в тензор
void col2im(float *matrix, Tensor *tensor, int ker_size);
// matrix - исходная матрица
// tensor - тензор, который нужно заполнить
// ker_size - размер ядра

#endif