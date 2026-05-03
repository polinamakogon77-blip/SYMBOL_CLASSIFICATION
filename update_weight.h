#ifndef UPDATE_WEIGHT
#define UDATE_WEIGHT


// ищет градиенты
void update_weight(float *gradient_in, float *full_conv, float *full_conv_filter, float **gradient_out, \
    float **g_weight, float **g_bias, int count_pictures, int size_input); 
// gradient_in - градиент soft_max
// full_conv - вход полносвязного слоя, т.е результат 2-ого сверточного слоя
// full_conv_filter - веса полносвязного слоя
// gradient_out, g_weight, g_bias - градиенты для 2-ого сверточного слоя, весов, смещения
// count_pictures - к-во изображений в тензоре
// size_input - размер одного изображения при входе в полносвязный слой


#endif