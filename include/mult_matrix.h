#ifndef MULT_MATRIX_H
#define MULT_MATRIX_H


// умножение матрицы ядра(все ядра объединены в одну матрицу) на матрицу изображения
float *mult_matrix(float *matrix, float *ker, int cnt_row_matrix, int cnt_col_matrix, int cnt_row_ker, int cnt_col_ker);
// matrix - матрица изображения 
// ker - матрица ядер
// cnt_row_matrix - к-во строк в матрице
// cnt_col_matrix - к-во рядов в матрице
// cnt_row_ker - к-во строк в ядре
// cnt_col_ker - к-во столбцов в ядре

#endif