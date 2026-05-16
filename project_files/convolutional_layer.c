#include <stdlib.h>
#include <string.h>
#include "convolutional_layer.h"
#include "im2col.h"
#include "mult_matrix.h"
#include "pooling.h"
#include "relu.h"
#include "col2im.h"


float *conv_layer(Tensor *tensor, float *ker, int **index, float *bias, int size_ker, int row_ker, int col_ker, int *pool_row, int *pool_col, float **input_relu, float **matrix_col) {

    int row_res, col_res; // размер результата
    float *matrix = im2col(tensor, size_ker, &row_res, &col_res);
    
    // запоминаю результат im2col
    *matrix_col = (float*)malloc(sizeof(float) * row_res * col_res); 
    memcpy(*matrix_col, matrix, sizeof(float) * row_res * col_res);

    matrix = mult_matrix(matrix, ker, row_res, col_res, row_ker, col_ker);
    // размер для polling
    int feature_h = (tensor->height + 2 - size_ker) + 1;
    int feature_w = (tensor->width + 2 - size_ker) + 1;

    // добавляем смещение для каждого элемента
    for (int i = 0; i < row_ker; ++i) {
        for (int j = 0; j < feature_h * feature_w; ++j) matrix[i * feature_h * feature_w + j] += bias[i];
    }

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

    return all_pooled;
}

void gradient_conv(float *gradient_in, Tensor *tensor, float *weight, float *matrix_col, int ker_size, int cnt_filter, float *grad_weight, \
    float *grad_bias, float *grad_input) {
    
    // размер одного изображения
    int h = tensor->height;
    int w = tensor->width;
    // размер одного изображения с рамкой
    int pad_h = h + 2;
    int pad_w = w + 2;
    // размер одного изображения после im2col с учетом каналов
    int h_im = tensor->count_channel * ker_size * ker_size; 
    int w_im = (pad_h - ker_size + 1) * (pad_w - ker_size + 1);

    // градиент для смещения
    for (int f = 0; f < cnt_filter; ++f) { // по всем фильтрам
        grad_bias[f] = 0;
        for (int i = 0; i < tensor->count_picture; ++i) { // по всем изображениям
            for (int col = 0; col < w_im; ++col) { // по всем столбцам изображения
                grad_bias[f] += gradient_in[(i * cnt_filter + f) * w_im + col]; // складываем ошибку
            }
        }
        
    }

    // градниент для весов
    // grad_weight = gradient_in * matrix_col.T
    for (int i = 0; i < tensor->count_picture; ++i) { // по всем изображениям
        for (int f = 0; f < cnt_filter; ++f) { // по всем фильтрам
            for (int row = 0; row < h_im; ++row) { // по строкам
                float sum = 0;
                for (int col = 0; col < w_im; ++col) { // по столбцам
                    sum += gradient_in[i * cnt_filter * w_im + f * w_im + col] * matrix_col[i * h_im * w_im + row * w_im + col];
                }  
                grad_weight[f * h_im + row] += sum;
            } 
        }
    }

    // градиент для входа
    // grad_col = weight.T * gradient_in
    float *weight_T = (float *)malloc(sizeof(float) * cnt_filter * h_im);
    // транспонируем м-цу весов
    for (int i = 0; i < cnt_filter; ++i) {
        for (int j = 0; j < h_im; ++j) {
            weight_T[j * cnt_filter + i] = weight[i * h_im + j];
        }
    }

    float *grad_col = (float *)calloc(tensor->count_picture * h_im * w_im, sizeof(float)); // градиент выхода но в виде столбчатой м-цы 

    for (int i = 0; i < tensor->count_picture; ++i) {
        for (int k = 0; k < h_im; ++k) {
            for (int p = 0; p < w_im; ++p) {
                float sum = 0;
                for (int f = 0; f < cnt_filter; ++f) {
                    int grad_idx = (i * cnt_filter + f) * w_im + p;
                    sum += weight_T[k * cnt_filter + f] * gradient_in[grad_idx];
                }
                grad_col[i * h_im * w_im + k * w_im + p] = sum;
            }
        }
    }

    // создаем тензор для grad_input
    Tensor *temp = create_tensor(NULL, tensor->count_picture, tensor->count_channel, h, w);
    temp->data = (float *)malloc(sizeof(float) * h * w * tensor->count_channel * tensor->count_picture);
    col2im(grad_col, temp, ker_size);
    memcpy(grad_input, temp->data, sizeof(float) * h * w * tensor->count_channel * tensor->count_picture);
    
    free_tensor(temp);
    free(grad_col);
    free(weight_T);

}