import numpy as np
import time
from scipy.linalg import blas as scipy_blas
from numpy.linalg import multi_dot


def generate_random_complex_matrix(size):
    return np.random.rand(size, size) + 1j * np.random.rand(size, size)


def matrix_multiply_algebr(A, B):
    return np.dot(A, B)


def matrix_multiply_cblas(A, B):
    return scipy_blas.cgemm(1.0, A, B)


def matrix_multiply_optimized(A, B):
    # Перемножение матриц с использованием np.matmul
    return np.matmul(A, B)


def measure_performance(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


# Размер матрицы
matrix_size = 1024

# Генерация матриц
matrix_A = generate_random_complex_matrix(matrix_size)
matrix_B = generate_random_complex_matrix(matrix_size)

# Перемножение матриц по различным методам и измерение времени выполнения
result_naive, time_naive = measure_performance(matrix_multiply_algebr, matrix_A, matrix_B)
result_cblas, time_cblas = measure_performance(matrix_multiply_cblas, matrix_A, matrix_B)
result_optimized, time_optimized = measure_performance(matrix_multiply_optimized, matrix_A, matrix_B)

# Оценка производительности в MFlops
c = 2 * matrix_size ** 3
mflops_algebr = c / (time_naive * 10 ** -6)
mflops_cblas = c / (time_cblas * 10 ** -6)
mflops_optimized = c / (time_optimized * 10 ** -6)

# Вывод результатов
print("Размерность матрицы:", matrix_size)
print("Время выполнения (алгебраический метод):", time_naive, "сек")
print("Время выполнения (cblas):", time_cblas, "сек")
print("Время выполнения (оптимизированный метод):", time_optimized, "сек")
print("Производительность (алгебраический метод):", mflops_algebr, "MFlops")
print("Производительность (cblas):", mflops_cblas, "MFlops")
print("Производительность (оптимизированный метод):", mflops_optimized, "MFlops")