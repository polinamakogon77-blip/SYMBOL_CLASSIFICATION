#include <stdio.h>
#include <stdlib.h>
#include "mult_matrix.h"


float *mult_matrix(float *matrix, float *ker, int cnt_row_matrix, int cnt_col_matrix, int cnt_row_ker, int cnt_col_ker) {
    // массив для записи результата
    float *res = (float*)calloc(cnt_row_ker * cnt_col_matrix, sizeof(float));

    for (int i = 0; i < cnt_row_ker; ++i) {
        for (int k = 0; k < cnt_col_ker; ++k) {
            int ker_i = ker[i * cnt_col_ker + k];

            if (ker_i == 0) continue; // пропускае нули

            for (int j = 0; j < cnt_col_matrix; ++j) {
                res[i * cnt_col_matrix + j] += ker_i * matrix[k * cnt_col_matrix + j]; // обновление результата
            }
        }
    }

    return res;
}
