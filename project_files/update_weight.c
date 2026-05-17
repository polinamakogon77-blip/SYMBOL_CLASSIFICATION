#include <stdio.h>
#include <stdlib.h>
#include "update_weight.h"


void update(float *weight, float *gradient, int size, float alpha) {
    for (int i = 0; i < size; ++i) {
        float grad = gradient[i];
        // Ограничиваем градиент
        if (grad > 1.0f) grad = 1.0f;
        if (grad < -1.0f) grad = -1.0f;
        weight[i] -= grad * alpha;
    }
}