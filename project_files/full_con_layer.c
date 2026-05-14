#include <stdlib.h>
#include "mult_matrix.h"


#define SIZE_ALF 10


float *fullconnected_layer(float *vector, float *weight, float *displacement, int size_vec) {
    float *result = mult_matrix(vector, weight, size_vec, 1, SIZE_ALF, size_vec);

    for (int i = 0; i < SIZE_ALF; ++i) result[i] += displacement[i];

    return result;

}

void gradient_full(float *gradient_in, float *full_conv, float *full_conv_filter, \
    float **gradient_out, float **g_weight, float **g_bias, int count_pictures, int size_input) {

        // выделяем память
        *gradient_out = calloc(size_input * count_pictures, sizeof(float));
        *g_weight = calloc(size_input * 10, sizeof(float));
        *g_bias = calloc(10, sizeof(float));
        
        for (int i = 0; i < count_pictures; ++i) {
            float *picture_gradient = gradient_in + i * 10; // градиент текущего изображения
            float *picture_full_conv = full_conv + i * size_input; // текущее изображение

            for (int j = 0; j < 10; ++j) {
                (*g_bias)[j] += picture_gradient[j]; // градиент смещения

                for (int k = 0; k < size_input; ++k) {
                    (*g_weight)[j * size_input + k] += picture_gradient[j] * picture_full_conv[k]; // гдадиент весов
                }
            }
        }

        for (int i = 0; i < count_pictures; ++i) {
            float *picture_gradient = gradient_in + i * 10; // градиент текущего изображения
            float *picture_res = (*gradient_out) + i * size_input; // результат градиента для текущего изображения \
            (куда записать результат)

            for (int j = 0; j < 10; ++j) {
                for (int k = 0; k < size_input; ++k) {
                    picture_res[k] += picture_gradient[j] * (*g_weight)[j * size_input + k]; // градиент для второго сверточного слоя

                }
            }
        }
    }