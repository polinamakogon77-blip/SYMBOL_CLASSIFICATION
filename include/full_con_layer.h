#ifndef FULL_CON_LAYER
#define FULL_CON_LAYER


//реализует полносвязный слой
float *fullconnected_layer(float *vecotor, float *weight, float *displacement, int size_vec);
// vector - вектор (получен из 2-ого сверточного слоя, размер n * 1)
// weight - матрица весов (26 * n)
// displacement - смещение для каждой буквы
// size_vec - размер вектора

#endif