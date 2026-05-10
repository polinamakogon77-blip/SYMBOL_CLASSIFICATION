#include <stdlib.h>
#include <string.h>
#include "convolutional_layer.h"
#include "im2col.h"
#include "mult_matrix.h"
#include "pooling.h"
#include "relu.h"


float *conv_layer(Tensor *tensor, float *ker, int **index, int size_ker, int row_ker, int col_ker, int *pool_row, int *pool_col, float **input_relu) {

    int row_res, col_res; // размер результата
    float *matrix = im2col(tensor, size_ker, &row_res, &col_res);
    matrix = mult_matrix(matrix, ker, row_res, col_res, row_ker, col_ker);
    // размер для polling
    int feature_h = (tensor->height + 2 - size_ker) + 1;
    int feature_w = (tensor->width + 2 - size_ker) + 1;

    //запоминаю матрицу, которая подается на вход в ReLU
    *input_relu = (float *)malloc(sizeof(float) * feature_h * feature_w * row_ker);
    memcpy(*input_relu, matrix, sizeof(float) * feature_h * feature_w * row_ker);

    ReLU(matrix, row_ker * feature_h, feature_w);

    // выделяем память для массива индексов
    int size_arr_index = my_ceil(feature_h, 2) * my_ceil(feature_w, 2); // размер массива индексов для одного фильтра
    *index = (int *)malloc(sizeof(int) * row_ker * size_arr_index);

    // MaxPool для каждого фильтра
    float *all_pooled = (float*)malloc(row_ker * (feature_h/2) * (feature_w/2) * sizeof(float));

    for (int i = 0; i < row_ker; ++i) { 
        float *filter = matrix + i * (feature_h * feature_w);  // указатель на i-й фильтр 
        // MaxPool
        int *index_filter; // времененный массив индексов для одного фильтра
        float *pooled = pooling(filter, &index_filter, feature_h, feature_w, pool_row, pool_col);
        
        // копируем результат
        for (int j = 0; j < (*pool_row) * (*pool_col); ++j) {
            all_pooled[i * (*pool_row) * (*pool_col) + j] = pooled[j];
        }
        for (int j = 0; j < size_arr_index; ++j) {
            (*index)[i * size_arr_index + j] = index_filter[j];
        }

        free(pooled);
        free(index_filter);
    }

    free(matrix);
    return all_pooled;
}
