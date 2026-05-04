#ifndef POOLING_H
#define POOLING_H

// уменьшае размеры матрицы
// разбивает матрицу на области 2 * 2
// в каждой области оставляет только максимальное значение
float *pooling(float *matrix, int **index, int row, int col, int *res_row, int *res_col);
// matrix - входная матрица
// index - индексы максимальных элементов
// row * col - размер матрицы
// res_row * res_col - размер результата

// градиент ф-ии пулирования
void *gradient_pooling(float *gradient_in, float *index, float *gradient_out, int size_in);
// gradient_in - градиент для 2-ого сверточного слоя, полученный из update_weight 
// index - индексы максимальных элементов
// gradient_out - результат
// size_in - размер входа


#endif