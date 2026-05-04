#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "tensor.h"


// реализует сверточный слой
float *conv_layer(Tensor *tensor, float *ker, int **index, int size_ker, int row_ker, int col_ker, int *pool_row, int *pool_col);
// tensor - набор изображений
// size_ker - размер ядра
// int row_ker, int col_ker - размер массива весов
// pool_row, pool_col - указатели на размеры результата
// index - индексы максисальных элементов

#endif