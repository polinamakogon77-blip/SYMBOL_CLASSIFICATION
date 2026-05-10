#include <stdlib.h>
#include "mult_matrix.h"


#define SIZE_ALF 10


float *fullconnected_layer(float *vector, float *weight, float *displacement, int size_vec) {
    float *result = mult_matrix(vector, weight, size_vec, 1, SIZE_ALF, size_vec);

    for (int i = 0; i < SIZE_ALF; ++i) result[i] += displacement[i];

    return result;

}