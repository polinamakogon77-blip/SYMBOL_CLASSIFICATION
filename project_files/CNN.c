#include <stdlib.h>
#include "convolutional_layer.h"
#include "full_con_layer.h"
#include "soft_max.h"
#include "update_weight.h"
#include "pooling.h"
#include "relu.h"
#include <math.h>


float *CNN(Tensor *tensor, float *filter_1, float *filter_2, float *filter_3, float *displacement, int size_ker, int row_filter_1, int col_filter_1, int *res_size, float *loss) {
    // ===========FORWARD PASS===========

    // первый сверточный слой
    int *index_poll_1; // массив индексов макимальных элементов для 1-го слоя
    int row_conv_1, col_conv_1; // размер результата
    float *input_relu_1; // матрица, которая подается на вход ф-ии активации
    float *conv_1 = conv_layer(tensor, filter_1, &index_poll_1, size_ker, row_filter_1, col_filter_1, &row_conv_1, &col_conv_1, &input_relu_1);

    //создание тензора из результата 1-го слоя
    Tensor *tensor_conv = create_tensor(conv_1, row_conv_1, col_conv_1, row_filter_1, tensor->count_picture);

    // второй сверточный слой
    int *index_poll_2; // массив индексов макимальных элементов для 2-го слоя
    int row_conv_2, col_conv_2; // размер результата
    float *input_relu_2; // матрица, которая подается на вход ф-ии активации
    float *conv_2 = conv_layer(tensor_conv, filter_2, &index_poll_2, size_ker, row_filter_1 * 2, size_ker * size_ker * row_filter_1, &row_conv_2, &col_conv_2, &input_relu_2);

    *loss = 0; // оценивает точность предсказаний

    int size_picture = row_conv_2 * col_conv_2; // размер одного изображения
    float *full_conv = (float*)malloc(sizeof(float) * tensor->count_picture * 10);
    for (int i = 0; i < tensor->count_picture; ++i) {
        float *current_pictures = conv_2 + i * size_picture; // текущее изображение
        // полносвязный слой
        float *full_conv_picture = fullconnected_layer(current_pictures, filter_3, displacement, row_conv_2 * col_conv_2);

        // преобразуем знчения в вероятности
        soft_max(full_conv_picture);

        *loss += -logf(full_conv_picture[tensor->class_picture[i]] + 1e-8f);
        // запись в общий массив
        for (int j = 0; j < 10; ++j) full_conv[i * 10 + j] = full_conv_picture[j];
        free(full_conv_picture);
    }

    *res_size = tensor->count_picture * 10;
    *loss  /= tensor->count_picture;
    
    // ===========BACKWARD PASS===========
    soft_max_gradient(full_conv, tensor);
    
    // ====градиенты для 2-ого сверточного слоя====
    // вычисляем градиенты
    float *gradient_out;
    float *gradient_weight;
    float *gradient_bias;
    update_weight(full_conv, conv_2, filter_3, &gradient_out, &gradient_weight, &gradient_bias, tensor->count_picture, row_conv_2 * col_conv_2);

    // градиент для ф-ии пулирования
    float *gradient_poll;
    gradient_pooling(gradient_out, index_poll_2, &gradient_poll, tensor->count_picture * row_conv_2 * col_conv_2);    

    //градиент для ф-ии активации
    float *gradient_relu;
    gradient_ReLU(gradient_poll, input_relu_2, &gradient_relu, tensor->count_picture * row_conv_2 * col_conv_2 * 4);

    // очистка
    free(conv_1);
    free_tensor(tensor_conv);
    free(conv_2);
    free(index_poll_1);
    free(index_poll_2);
    free(gradient_out);
    free(gradient_weight);
    free(gradient_bias);
    free(gradient_poll);
    free(gradient_relu);
    free(input_relu_1);
    free(input_relu_2);

    return full_conv;
}