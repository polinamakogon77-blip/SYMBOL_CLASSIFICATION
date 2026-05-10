#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>


// один тензор - максимум 128 изображений
typedef struct Tensor { 
    float *data; // значения
    int count_picture; // к-во изображений
    int *class_picture; // классы изображений
    int count_channel; // к-во каналов
    int height; // высота изображения
    int width; // ширина изображения
    int step_picture; // шаг по изображениям
    int step_channel; // шаг по каналам
    int step_height; // шаг в высоту
    int step_width; // шаг в ширину
}Tensor;


// инициализация тензора
Tensor *init_tensor(FILE *data, int picture, int channel, int h, int w);
// data - файл с данными
// picture - к-во изображений в тензоре
// channel - к-во каналов
// h, w - размер изображения

// создание тензора из массива
Tensor *create_tensor(float *data, int row, int col, int cnt_filter, int cnt_picture);
// data - массив с данными
// row, col - размер массива
// cnt_filter - к-во фильтров на первом слое == к-во каналов на втором
// cnt_picture - к-во изображений в тензоре

// удаление тензора
void free_tensor(Tensor *tensor); 


#endif