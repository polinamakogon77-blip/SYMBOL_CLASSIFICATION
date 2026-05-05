#include <stdlib.h>
#include "relu.h"

void *ReLU(float *matrix, int row, int col) {
    for (int i = 0; i < row * col; ++i) {
            matrix[i] = matrix[i] > 0? matrix[i]: 0; // меняет отрицательное значение на ноль   
    }
}

void gradient_ReLU(float *gradient_in, float *input, float **gradient_relu, int size) {
    *gradient_relu = (float *)malloc(sizeof(float) * size);

    for (int i = 0; i < size; ++i) {
        // если вход отрицательный - градиент = 0
        // иначе - градиент = градиенту_pooling
        (*gradient_relu)[i] = input[i] < 0? 0: gradient_in[i];
    }
}