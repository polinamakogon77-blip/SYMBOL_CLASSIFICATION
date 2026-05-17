#include <math.h>
#include "soft_max.h"


#define SIZE_ALF 10


void soft_max(float *vector) {
    // ищет максимальное значение
    float mx = vector[0]; 
    for (int i = 0; i < SIZE_ALF; ++i) mx = mx > vector[i]? mx: vector[i];

    // вычитает из каждого значения максимальное
    // вычисляет экспонениту
    // находит сумму экспонент
    float summa = 0;
    for (int i = 0; i < SIZE_ALF; ++i) {
        vector[i] = expf(vector[i] - mx);
        summa += vector[i];
    }

    // нормирует значения
    for (int i = 0; i < SIZE_ALF; ++i) {
        vector[i] = vector[i] / summa;
        if (vector[i] < 1e-8f) vector[i] = 1e-8f;
    }

}

void soft_max_gradient(float *batch, Tensor *tensor) {
    for (int i = 0; i < tensor->count_picture; ++i) {
        batch[i * 10 + tensor->class_picture[i]] -= 1;
    }
}