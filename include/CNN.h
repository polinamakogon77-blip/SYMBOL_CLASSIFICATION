#ifndef CNN_H
#define CNN_H

#include "tensor.h"


// принимает набор изображений
// для каждого изображения возвращате вероятность равентсва с буквой 
float *FORWARD_PASS(Tensor *tensor, float *filter_1, float *filter_2, float *filter_3, float *bias_1, float *bias_2, float * bias_3, int size_ker, int row_filter_1, int col_filter_1, int *res_size, float *loss); 
// tensor - набор изображений
// filter_1, filter_2 - массивы фильтров для 1-ого и 2-ого сверточных слоев
// size_ker - размер ядра
// int row_filter_1, int col_filter_1 - размер массива весов для 1-ого слоя
// res_size размер массива с вероятностями
// bias_1, bias_2, bias_3 - смещения
// loss - ошибка

#endif