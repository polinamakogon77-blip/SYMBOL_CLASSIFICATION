#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"


Tensor *tensor_init(FILE *data, int picture, int channel, int h, int w) {
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->count_picture = picture;
    tensor->count_channel = channel;
    tensor->height = h;
    tensor->width = w;

    tensor->data = (float *)calloc(picture * h * w * channel, sizeof(float));
    tensor->class_picture = (int*)malloc(sizeof(int) * picture);

    // заполнение массива
    for (int i = 0; i < picture; ++i) {
        int offset = i * h * w * channel;

        fread(tensor->data + offset, sizeof(float), h * w * channel, data);
        fread(&tensor->class_picture[i], sizeof(int), 1, data);
    }

    // подсчет шагов
    tensor->step_picture = w * h * channel;
    tensor->step_channel = w * h;
    tensor->step_height = w;
    tensor->step_width = 1;

    return tensor;
}