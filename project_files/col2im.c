#include <stdlib.h>
#include "col2im.h"


void col2im(float *matrix, Tensor *tensor, int ker_size) {
    // размер одного изображения с рамкой
    int padding_width = tensor->width + 2;
    int padding_height = tensor->height + 2;

    // размер одного изображения на входе
    int in_width = (padding_width - ker_size + 1) * (padding_height - ker_size + 1);
    int in_height = ker_size * ker_size;

    // временный массив, чтобы хранить результат с рамкой
    float *tensor_padding = (float*)calloc(padding_height * padding_width * tensor->count_picture * tensor->count_channel, sizeof(float));

    for (int b = 0; b < tensor->count_picture * tensor->count_channel; ++b) { // по всем изображениям (каждый канал считать за отдельное изображение)
        for (int kh = 0; kh < ker_size; ++kh) {
            for (int kw = 0; kw < ker_size; ++kw) {
                int row = kh * ker_size + kw; // номер стрики
                
                for (int h = 0; h < padding_height - ker_size + 1; ++h) { // по всем позициам на вертикали
                    for (int w = 0; w < padding_width - ker_size + 1; ++w) { // по всем позициям на горизонтали
                        int col = (b * in_height + row) * in_width + h * (padding_width - ker_size + 1) + w; // индекс в входной матрице
                        int idx = b * padding_height * padding_width + (h + kh) * padding_width + w + kw; // куда поместить элемент

                        tensor_padding[idx] += matrix[col];
                    }
                }
            }
        }
    }

    // обрезание рамки
    for (int b = 0; b < tensor->count_picture * tensor->count_channel; ++b) { // по всем изображениям (каждый канал считать за отдельное изображение)
        for (int h = 0; h < tensor->height; ++h) {
            for (int w = 0; w < tensor->width; ++w) {
                int idx_pad = b * padding_height * padding_width + (h + 1) * padding_width + (w + 1); // откуда 
                int idx_res = b * tensor->width * tensor->height + h * tensor->width + w; // куда

                tensor->data[idx_res] = tensor_padding[idx_pad];
            }
        }
    }

    free(tensor_padding);
}