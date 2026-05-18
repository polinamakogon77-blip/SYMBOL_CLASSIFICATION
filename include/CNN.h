#ifndef CNN_H
#define CNN_H

#include "tensor.h"

#define L2_COEFF 0.0005  // коэффициент регуляризации
#define L2_LOSS 0 // сумма квадратов всех весов


// принимает набор изображений
// для каждого изображения возвращате вероятность равентсва с буквой 
void CNN(Tensor *tensor, float *filter_1, float *filter_2, float *filter_3, float *bias_1, float *bias_2, float * bias_3, \
    int size_ker, int row_filter_1, int col_filter_1, int *res_size, float *loss, float alpha); 
// tensor - набор изображений
// filter_1, filter_2 - массивы фильтров для 1-ого и 2-ого сверточных слоев
// size_ker - размер ядра
// int row_filter_1, int col_filter_1 - размер массива весов для 1-ого слоя
// bias_1, bias_2, bias_3 - смещения
// loss - ошибка
// alpha - шаг обучения

#endif