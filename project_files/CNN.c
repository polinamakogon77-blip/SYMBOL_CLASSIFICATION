#include <stdlib.h>
#include "convolutional_layer.h"
#include "full_con_layer.h"
#include "soft_max.h"
#include "update_weight.h"
#include "pooling.h"
#include "relu.h"
#include <math.h>


float *CNN(Tensor *tensor, float *filter_1, float *filter_2, float *filter_3, float *bias_1, float *bias_2, float *bias_3, int size_ker, int row_filter_1, int col_filter_1, int *res_size, float *loss) {
    // ===========FORWARD PASS===========

    // первый сверточный слой
    int *index_poll_1; // массив индексов макимальных элементов для 1-го слоя
    int row_conv_1, col_conv_1; // размер результата
    float *input_relu_1; // матрица, которая подается на вход ф-ии активации
    float *matrix_col_1; // результат im2col
    float *conv_1 = conv_layer(tensor, filter_1, &index_poll_1, bias_1, size_ker, row_filter_1, col_filter_1, &row_conv_1, &col_conv_1, &input_relu_1, &matrix_col_1);

    //создание тензора из результата 1-го слоя
    Tensor *tensor_conv = create_tensor(conv_1, row_conv_1, col_conv_1, row_filter_1, tensor->count_picture);

    // второй сверточный слой
    int *index_poll_2; // массив индексов макимальных элементов для 2-го слоя
    int row_conv_2, col_conv_2; // размер результата
    float *input_relu_2; // матрица, которая подается на вход ф-ии активации
    float *matrix_col_2; // результат im2col
    float *conv_2 = conv_layer(tensor_conv, filter_2, &index_poll_2, bias_2, size_ker, row_filter_1 * 2, size_ker * size_ker * row_filter_1, &row_conv_2, &col_conv_2, &input_relu_2, &matrix_col_2);

    *loss = 0; // оценивает точность предсказаний

    int size_picture = row_conv_2 * col_conv_2; // размер одного изображения
    float *full_conv = (float*)malloc(sizeof(float) * tensor->count_picture * 10);
    for (int i = 0; i < tensor->count_picture; ++i) {
        float *current_pictures = conv_2 + i * size_picture; // текущее изображение
        // полносвязный слой
        float *full_conv_picture = fullconnected_layer(current_pictures, filter_3, bias_3, row_conv_2 * col_conv_2);

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
    
    // вычисляем градиенты
    float *gradient_out;
    float *gradient_filter3;
    float *gradient_bias_3;
    update_weight(full_conv, conv_2, filter_3, &gradient_out, &gradient_filter3, &gradient_bias_3, tensor->count_picture, row_conv_2 * col_conv_2);

    // градиент для ф-ии пулирования_2
    float *gradient_poll2;
    gradient_pooling(gradient_out, index_poll_2, &gradient_poll2, tensor->count_picture * row_conv_2 * col_conv_2);    

    //градиент для ф-ии активации_2
    float *gradient_relu2;
    gradient_ReLU(gradient_poll2, input_relu_2, &gradient_relu2, row_filter_1 * 2 * (tensor_conv->height + 2 - size_ker + 1) * (tensor_conv->width + 2 - size_ker + 1));

    // градиент для 2-ого сверточного слоя
    float *gradient_conv2 = (float *)malloc(sizeof(float) * tensor_conv->count_channel * tensor_conv->count_picture * tensor_conv->height * tensor_conv->width);
    float *gradient_filter2 = (float *)calloc(tensor_conv->count_channel * size_ker * size_ker, sizeof(float));
    float *gradient_bias_2 = (float *)calloc(row_filter_1 * 2, sizeof(float));
    gradient_conv(gradient_relu2, tensor_conv, filter_2, matrix_col_2, size_ker, row_filter_1 * 2, gradient_filter2, gradient_bias_2, gradient_conv2); 

     // градиент для ф-ии пулирования
    float *gradient_poll1;
    gradient_pooling(gradient_conv2, index_poll_1, &gradient_poll1, tensor->count_picture * row_conv_1 * col_conv_1);    

    //градиент для ф-ии активации
    float *gradient_relu1;
    gradient_ReLU(gradient_poll1, input_relu_1, &gradient_relu1, row_filter_1 * (tensor->height + 2 - size_ker + 1) * (tensor->width + 2 - size_ker + 1));

    // градиент для 1-ого сверточного слоя
    float *gradient_conv1 = (float *)malloc(sizeof(float) * tensor->count_channel * tensor->count_picture * tensor->height * tensor->width);
    float *gradient_filter1 = (float *)calloc(tensor->count_channel * size_ker * size_ker, sizeof(float));
    float *gradient_bias_1 = (float *)calloc(row_filter_1, sizeof(float));
    gradient_conv(gradient_relu1, tensor, filter_1, matrix_col_1, size_ker, row_filter_1, gradient_filter1, gradient_bias_1, gradient_conv1); 


    // очистка
    free(conv_1);
    free_tensor(tensor_conv);
    free(conv_2);
    free(index_poll_1);
    free(index_poll_2);
    free(gradient_out);
    free(gradient_poll1);
    free(gradient_relu1);
    free(gradient_poll2);
    free(gradient_relu2);
    free(input_relu_1);
    free(input_relu_2);
    free(gradient_conv1);
    free(gradient_conv2);

    // временно
    free(gradient_bias_1);
    free(gradient_bias_2);
    free(gradient_bias_3);
    free(gradient_filter1);
    free(gradient_filter2);
    free(gradient_filter3);

    return full_conv;
}