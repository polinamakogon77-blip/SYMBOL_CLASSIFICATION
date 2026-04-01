#include <stdio.h>


// один тензор - максимум 128 изображений
typedef struct Tensor { 
    float *data; // значения
    int count_picture; // к-во изображений
    int *class_picture; // классы изображений
    int count_channel; // к-во каналов
    int height; // высота изображения
    int wight; // ширина изображения
    int step_picture; // шаг по изображениям
    int step_channel; // шаг по каналам
    int step_height; // шаг в высоту
    int step_wight; // шаг в ширину
}Tensor;


Tensor *init_tensor(FILE *data, int picture, int channel, int h, int w); // инициализация тензора
void free_tensor(Tensor *tensor); // удаление тензора