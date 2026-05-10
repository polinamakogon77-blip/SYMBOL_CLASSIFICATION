#include <stdlib.h>
#include "tensor.h"


// читает бинарный файл с изображениями
// создает массив указателей на тензоры
Tensor **read_file(FILE *file, int *count_tensor) {
    // вычисляем к-во изображений в файле
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    int count_bytes = sizeof(int) + 784 * sizeof(float);
    int count_pictutes = file_size / count_bytes;
    *count_tensor = (count_pictutes + 127) / 128;

    Tensor **array = (Tensor**)malloc(sizeof(Tensor*) * (*count_tensor));

    // заполнение массива
    int index = 0; // индекс указателя тензора в массиве
    while (count_pictutes > 0) {
        fseek(file, index * 128 * count_bytes, SEEK_SET); // следующий батч
        array[index] = init_tensor(file, count_pictutes < 128? count_pictutes: 128, 1, 28, 28);
        count_pictutes -= 128;
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
    // чтение файла
    FILE *data = fopen("notMinist.bin", "rb");
    int count_tensor;
    Tensor **array = read_file(data, &count_tensor);

    



    // очистка
    for (int i = 0; i < count_tensor; ++i) free_tensor(array[i]);
    free(array);

    fclose(data);
    return 0;
}