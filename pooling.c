#include <stdlib.h>
#include "pooling.h"


// делит число a на b с округлением вверх
int my_ceil(int a, int b) {
    if (a % b == 0) return a / b;
    return a / b + 1;
}

// ищет максимальное число
float max(float a, float b, float c, float d) {
    float mx = a > b? a: b;
    mx = mx > c? mx: c;
    mx = mx > d? mx: d;

    return mx;
}


float *pooling(float *matrix, int row, int col, int *res_row, int *res_col) {
    // матрица результата
    *res_row = my_ceil(row, 2);
    *res_col = my_ceil(col, 2);
    float *result = (float *)malloc(sizeof(float) * (*res_col) * (*res_row));

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

            result[i * (*res_col) + j] = max(a, b, c, d);

        }
    }

    return result;

}