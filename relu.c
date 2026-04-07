#include <stdlib.h>
#include "relu.h"

void ReLU(float *matrix, int row, int col) {
    for (int i = 0; i < row * col; ++i) {
            matrix[i] = matrix[i] > 0? matrix[i]: 0; // меняет отрицательное значение на ноль   
    }
}