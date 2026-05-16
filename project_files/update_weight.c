#include <stdio.h>
#include <stdlib.h>
#include "update_weight.h"


void update(float *weight, float *gradient, int size, float alpha) {
    for (int i = 0; i < size; ++i) {
        weight[i] -= gradient[i] * alpha;
    }
}