#ifndef RELU_H
#define RELU_H


// ReLU - ф-ия активации - устанавливает отрицательные значения в нуль 
void ReLU(float *matrix, int row, int col);
// matrix - входная матрица 
// row * col - размер м-цы

// градиент для ReLU
void gradient_ReLU(float *gradient_in, float *input, float **gradient_relu, int size);
// gradient_in - градиент от ф-ии пулирования  
// input - вход, который подавался ReLU в прямом обходе
// gradient_relu - результат
// size - размер входа/выхода

#endif