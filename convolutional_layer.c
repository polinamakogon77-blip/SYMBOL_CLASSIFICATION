#include <stdlib.h>
#include "im2col.h"
#include "mult_matrix.h"
#include "pooling.h"
#include "relu.h"

float *conv_layer(Tensor *tensor, float *ker, int size_ker, int row_ker, int col_ker, int *pool_row, int *pool_col) {

    int row_res, col_res; // размер результата
    float *matrix = im2col(tensor, size_ker, &row_res, &col_res);
    matrix = mult_matrix(matrix, ker, row_res, col_res, row_ker, col_ker);
    // размер для polling
    int feature_h = (tensor->height + 2 - size_ker) + 1;
    int feature_w = (tensor->width + 2 - size_ker) + 1;
    ReLU(matrix, row_ker * feature_h, feature_w);

    // MaxPool для каждого фильтра
    float *all_pooled = (float*)malloc(row_ker * (feature_h/2) * (feature_w/2) * sizeof(float));

    for (int i = 0; i < row_ker; ++i) { 
        float *filter = matrix + i * (feature_h * feature_w);  // указатель на i-й фильтр 
        // MaxPool
        float *pooled = pooling(filter, feature_h, feature_w, pool_row, pool_col);
        
        // копируем результат
        for (int j = 0; j < (*pool_row) * (*pool_col); ++j) {
            all_pooled[i * (*pool_row) * (*pool_col) + j] = pooled[j];
        }
        free(pooled);
    }

    free(matrix);
    return all_pooled;
}