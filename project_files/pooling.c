#include <stdlib.h>
#include "pooling.h"


// делит число a на b с округлением вверх
int my_ceil(int a, int b) {
    if (a % b == 0) return a / b;
    return a / b + 1;
}

// ищет максимальное число
float max(float a, float b, float c, float d, int **index, int idx, int idx_a, int idx_b, int idx_c, int idx_d) {
    float mx = a;
    (*index)[idx] = idx_a;

    if (mx < b) {
        mx = b;
        (*index)[idx] = idx_b;
    }
    if (mx < c) {
        mx = c;
        (*index)[idx] = idx_c;
    }
    if (mx < d) {
        mx = d;
        (*index)[idx] = idx_d;
    }
    return mx;
}


float *pooling(float *matrix, int **index, int row, int col, int *res_row, int *res_col) {
    // матрица результата
    *res_row = my_ceil(row, 2);
    *res_col = my_ceil(col, 2);
    float *result = (float *)malloc(sizeof(float) * (*res_col) * (*res_row));

    *index = (int *)malloc(sizeof(int) * (*res_col) * (*res_row)); 

    for (int i = 0; i < *res_row; ++i) {
        for (int j = 0; j < *res_col; ++j) {
            // первый элемент области в матрице
            int row_1 = i * 2;
            int col_1 = j * 2;
            float a = matrix[row_1 * col + col_1]; 

            // второй элемент области в матрице
            float b = (col_1 + 1 >= col)? 0: matrix[row_1 * col + col_1+1];

            // третий элемент области в матрице
            float c = (row_1 + 1 >= row)? 0: matrix[(row_1 + 1) * col + col_1];

            // четвертый элемента области в матрице
            float d = (col_1 + 1 >= col) || (row_1 + 1 >= row)? 0: matrix[(row_1 + 1) * col + col_1+1];

            result[i * (*res_col) + j] = max(a, b, c, d, index, i * (*res_col) + j, row_1 * col + col_1, \
            row_1 * col + col_1+1, (row_1 + 1) * col + col_1, (row_1 + 1) * col + col_1+1);

        }
    }

    return result;

}

void gradient_pooling(float *gradient_in, int *index, float **gradient_out, int size_in) {
    *gradient_out = (float *)calloc(size_in * 4, sizeof(float));

    for (int i = 0; i < size_in * 4; ++i) {
        int idx = index[i];
        if (idx >= 0 && idx < size_in * 4) {
            (*gradient_out)[idx] += gradient_in[i];
        }
    }
}