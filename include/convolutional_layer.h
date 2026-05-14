#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "tensor.h"


// реализует сверточный слой
float *conv_layer(Tensor *tensor, float *ker, int **index, float *bias, int size_ker, int row_ker, int col_ker, int *pool_row, int *pool_col, float **input_relu, float **matrix_col);
// tensor - набор изображений
// size_ker - размер ядра
// int row_ker, int col_ker - размер массива весов
// pool_row, pool_col - указатели на размеры результата
// index - индексы максисальных элементов
// input_relu - матрица, которая подается на вход ф-ии активации
// matrix_col - результат im2col
// bias - смещение

// вычисляет градиент для сверточного слоя
void gradient_conv(float *gradient_in, Tensor *tensor, float *weight, float *matrix_col, int ker_size, int cnt_filter, float *grad_weight, \
    float *grad_bias, float *grad_input);
// gradient_in - градиент ReLU
// tensor - вход в im2col
// weight - веса фильтров
// matrix_col - результат im2col
// cnt_filter - к-во фильтров
// grad_weight, grad_bias, grad_input - градинеты весов, смещения, входа

#endif