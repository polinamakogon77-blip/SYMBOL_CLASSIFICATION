#include <stdlib.h>
#include "convolutional_layer.h"


float *CNN(Tensor *tensor, float *filter_1, float *filter_2, int size_ker, int row_filter_1, int col_filter_1, int *res_size) {
    // первый сверточный слой
    int row_conv_1, col_conv_1; // размер результата
    float *conv_1 = conv_layer(tensor, filter_1, size_ker, row_filter_1, col_filter_1, &row_conv_1, &col_conv_1);

    //создание тензора из результата 1-го слоя
    Tensor *tensor_conv = create_tensor(conv_1, row_conv_1, col_conv_1, row_filter_1, tensor->count_picture);

    // второй сверточный слой
    int row_conv_2, col_conv_2; // размер результата
    float *conv_2 = conv_layer(tensor_conv, filter_2, size_ker, row_filter_1 * 2, size_ker * size_ker * row_filter_1, &row_conv_2, &col_conv_2);

    free(conv_1);
    free(tensor_conv);
}