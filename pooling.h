#ifndef POOLING_H
#define POOLING_H

// уменьшае размеры матрицы
// разбивает матрицу на области 2 * 2
// в каждой области оставляет только максимальное значение
float *pooling(float *matrix, int row, int col, int *res_row, int *res_col);
// matrix - входная матрица
// row * col - размер матрицы
// res_row * res_col - размер результата

#endif