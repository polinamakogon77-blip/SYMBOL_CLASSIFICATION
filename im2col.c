#include <stdlib.h>
#include <string.h>
#include "im2col.h"


float *im2col(Tensor *tensor, int ker, int *res_row, int *res_col) {
    
    // добаление рамки из нулей, чтобы обработать края
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

    // каждое изображение на выходе имеет следующие размеры
    // высота = ker^2
    // ширина = (adv_matrix_height - ker + 1) * (adv_matrix_width - ker + 1)
    int adv_matrix_height = tensor->height + 2;
    int adv_matrix_width = tensor->width + 2; 
    *res_row = tensor->count_picture * tensor->count_channel * ker * ker;
    *res_col = (adv_matrix_height - ker + 1) * (adv_matrix_width - ker + 1);
    float *res = (float *)malloc(sizeof(float) * (*res_col) * (*res_row));

    // разбить изображение на блоки по ядру
    // выровнять их в столбцы
    int h_res = adv_matrix_height - ker + 1;
    int w_res = adv_matrix_width - ker + 1;
    int col_stride = h_res * w_res;

    for (int i = 0; i < tensor->count_picture; ++i) {
        for (int j = 0; j < tensor->count_channel; ++j) {
            for (int y = 0; y < ker; ++y) {
                for (int x = 0; x < ker; ++x) {
                    int row = y * ker + x; // индекс строки
                    
                    for (int h = 0; h < h_res; ++h) {
                        for (int w = 0; w < w_res; ++w) {
                            int col = h * w_res + w; // индекс столбца
                            
                            int src = (i * tensor->count_channel + j) * adv_matrix_height * adv_matrix_width \
                             + (h + y) * adv_matrix_width + (w + x);
                            int dst = (i * tensor->count_channel + j) * ker * ker * col_stride \
                             + row * col_stride + col;
                            
                            res[dst] = advanced_matrix[src];
                        }
                    }
                }
            }
        }
    }

    free(line);
    free(advanced_matrix);

    return res;
}
