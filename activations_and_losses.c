#include <math.h>


#define SIZE_ALF 10


void soft_max(float *vector) {
    // ищет максимальное значение
    float mx = vector[0]; 
    for (int i = 0; i < SIZE_ALF; ++i) mx = mx > vector[i]? mx: vector[i];

    // вычитает из каждого значения максимальное
    // вычисляет экспонениту
    // находит сумму экспонент
    float summa = 0;
    for (int i = 0; i < SIZE_ALF; ++i) {
        vector[i] = expf(vector[i] - mx);
        summa += vector[i];
    }

    // нормирует значения
    for (int i = 0; i < SIZE_ALF; ++i) vector[i] = vector[i] / summa;

}
