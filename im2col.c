#include <stdlib.h>
#include <string.h>
#include "im2col.h"


void im2col(Tensor *tensor, int ker, float *res) {
    
    // добление рамки из нулей, чтобы обработать края
    float *advanced_matrix = (float*)calloc(tensor->count_channel * tensor->count_picture * \
        (tensor->height + 2) * (tensor->width + 2), sizeof(float));
    float *line = (float*)calloc(tensor->width + 2, sizeof(float));
    for (int i = 0; i < tensor->count_picture; ++i) { // для каждой картинки
        for (int j = 0; j < tensor->count_channel; ++j) { // для каждого канала
            int offset = (i * tensor->count_channel + j) * (tensor->height + 2) * (tensor->width + 2);
            memcpy(advanced_matrix + offset, line, sizeof(float) * (tensor->width + 2)); // верхняя граница рамки

            for (int k = 0; k < tensor->height; ++k) { // для каждой строки изображения
                int offset_src = tensor->step_picture * i + tensor->step_channel * j + tensor->step_height * k; 
                memcpy(line + 1, tensor->data + offset_src, sizeof(float) * tensor->width); 
                memcpy(advanced_matrix + offset + (k + 1) * (tensor->width + 2), line, sizeof(float) * (tensor->width + 2)); // левая и правая границы рамки

            }

            memset(line, 0, (tensor->width + 2) * sizeof(float));
            memcpy(advanced_matrix + offset + (tensor->height + 1) * (tensor->width + 2), line, sizeof(float) * (tensor->width + 2)); // нижняя граница рамки
        }
        
    }
    free(line);
}