#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "tensor.h"
#include "CNN.h"


#define ALF 10 // к-во классов
#define CHANNEL 1 // к-во каналов


// читает бинарный файл с изображениями
// создает массив указателей на тензоры
Tensor **read_file(FILE *file, int *count_tensor) {
    // читаем заголовок
    int total_pictures;
    fread(&total_pictures, sizeof(int), 1, file);
    
    *count_tensor = (total_pictures + 15) / 16;
    Tensor **array = (Tensor**)malloc(sizeof(Tensor*) * (*count_tensor));
    
    int remaining = total_pictures;
    int index = 0;
    while (remaining > 0) {
        int batch_size = remaining < 16 ? remaining : 16;
        array[index] = init_tensor(file, batch_size, 1, 28, 28);
        
        remaining -= batch_size;
        index += 1;
    }
    
    return array;
}


// перемешивает массив тензоров
void suffle_tensor(Tensor **arr, int size) {
    for (int i = size - 1; i > 0; --i) {
        int idx = rand() % (i + 1); // случайный элемент из тех, что идут раньше i

        // меняем местами
        Tensor *temp = arr[i];
        arr[i] = arr[idx];
        arr[idx] = temp;
    }
}


int main(void) {
    srand(time(NULL));

    // чтение файла
    FILE *data = fopen("data/notMnist.bin", "rb");
    int count_tensor;
    Tensor **array = read_file(data, &count_tensor);

    //=======ПАРАМЕТРЫ СЕТИ=======
    int chanel = 1;
    int ker_size = 2; // размер ядра 5 на 5
    int cnt_filter1 = 2; // к-во фильтров на первом сверточном слое
    int size_filter1 = ker_size * ker_size * CHANNEL; // размер одного фильтра
    int cnt_filter2 = cnt_filter1 * 2; // к фильтров на втором сверточном слое
    int size_filter2 = ker_size * ker_size * cnt_filter1; // размер одного фильтра 
    int size_filter3 = ALF * ker_size * ker_size * cnt_filter2; // размер фильтра для полносвязного слоя

    //=======ИНИЦИАЛИЗАЦИЯ ВЕСОВ=======
    float *filter1 = (float *)malloc(sizeof(float) * cnt_filter1 * size_filter1);
    for (int i = 0; i < cnt_filter1 * size_filter1; ++i) filter1[i] = ((float)rand() / RAND_MAX - 0.5) * 0.02;

    float *filter2 = (float *)malloc(sizeof(float) * cnt_filter2 * size_filter2);
    for (int i = 0; i < cnt_filter2 * size_filter2; ++i) filter2[i] = ((float)rand() / RAND_MAX - 0.5) * 0.02;

    float *filter3 = (float *)malloc(sizeof(float) * size_filter3);
    for (int i = 0; i < size_filter3; ++i) filter3[i] = ((float)rand() / RAND_MAX - 0.5) * 0.02;

    float *bias1 = (float*)malloc(sizeof(float) * cnt_filter1);
    for (int i = 0; i < cnt_filter1; ++i) bias1[i] = ((float)rand() / RAND_MAX - 0.5) * 0.02;

    float *bias2 = (float *)malloc(sizeof(float) * cnt_filter2);
    for (int i = 0; i < cnt_filter2; ++i) bias2[i] = ((float)rand() / RAND_MAX - 0.5) * 0.02;

    float *bias3 = (float *)calloc(ALF, sizeof(float));

    //=======ОБУЧЕНИЕ=======
    int epochs = 2; // к-во эпох
    float alpha = 0.0001; // шаг обучения

    for (int epoch = 0; epoch < epochs; ++epoch) {
        suffle_tensor(array, count_tensor); // перемешивание тензоров

        float avg_loss = 0; // средняя точность на эпохе

        for (int i = 0; i < count_tensor; ++i) { // вызываем CNN для каждого тензора
            float loss_batch; // точность на батче
            int s;
            float *k = CNN(array[i], filter1, filter2, filter3, bias1, bias2, bias3, ker_size, cnt_filter1, size_filter1, &s, &loss_batch, alpha);
            avg_loss += loss_batch;
            if (k != NULL) {
                free(k);
                k = NULL;
            }
            
        }

        avg_loss /= count_tensor;
        printf("точность на эпохе %d: %f\n", epoch + 1, avg_loss);

    }

    // очистка
    for (int i = 0; i < count_tensor; ++i) free_tensor(array[i]);
    free(array);

    free(filter1);
    free(filter2);
    free(filter3);
    free(bias1);
    free(bias2);
    free(bias3);
    
    fclose(data);
    return 0;
}