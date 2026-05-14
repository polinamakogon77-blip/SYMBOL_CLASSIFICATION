#ifndef FULL_CON_LAYER
#define FULL_CON_LAYER


//реализует полносвязный слой
float *fullconnected_layer(float *vecotor, float *weight, float *displacement, int size_vec);
// vector - вектор (получен из 2-ого сверточного слоя, размер n * 1)
// weight - матрица весов (26 * n)
// displacement - смещение для каждой буквы
// size_vec - размер вектора

// ищет градиент полносвязно слоя
void gradient_full(float *gradient_in, float *full_conv, float *full_conv_filter, float **gradient_out, \
    float **g_weight, float **g_bias, int count_pictures, int size_input); 
// gradient_in - градиент soft_max
// full_conv - вход полносвязного слоя, т.е результат 2-ого сверточного слоя
// full_conv_filter - веса полносвязного слоя
// gradient_out, g_weight, g_bias - градиенты для 2-ого сверточного слоя, весов, смещения
// count_pictures - к-во изображений в тензоре
// size_input - размер одного изображения при входе в полносвязный слой

#endif