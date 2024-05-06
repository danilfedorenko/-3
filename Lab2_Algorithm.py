import numpy as np
import time
from scipy.linalg import blas as scipy_blas
from numpy.linalg import multi_dot


def generate_random_complex_matrix(size):
    real_part = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    imag_part = np.random.rand(matrix_size, matrix_size).astype(np.float32) * 1j
    return (real_part + imag_part).astype(np.complex64)


def multiply_algebr(A, B):
    return np.dot(A, B)


def multiply_cblas(A, B):
    return scipy_blas.cgemm(1.0, A, B)


def multiply_matmul(A, B):
    # Перемножение матриц с использованием np.matmul
    return np.matmul(A, B)


def measure_performance(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

def compare_matrix(A, B, epsilon=1e-6):
    return np.allclose(A, B, atol=epsilon)


# Размер матрицы
matrix_size = 1024

# Генерация матриц
matrix_A = generate_random_complex_matrix(matrix_size)
matrix_B = generate_random_complex_matrix(matrix_size)

# Перемножение матриц по различным методам и измерение времени выполнения
C1, time_algebr = measure_performance(multiply_algebr, matrix_A, matrix_B)
C2, time_cblas = measure_performance(multiply_cblas, matrix_A, matrix_B)
C3, time_matmul = measure_performance(multiply_matmul, matrix_A, matrix_B)

# Оценка производительности в MFlops
c = 2 * matrix_size ** 3
mflops_algebr = c / (time_algebr * 10 ** -6)
mflops_cblas = c / (time_cblas * 10 ** -6)
mflops_matmul = c / (time_matmul * 10 ** -6)

# Вывод результатов
print("Размерность матрицы:", matrix_size)
print(matrix_A[10][20])
print(matrix_B[10][20])
print("Время выполнения (алгебраический метод):", time_algebr, "сек")
print("Время выполнения (cblas):", time_cblas, "сек")
print("Время выполнения (оптимизированный метод):", time_matmul, "сек")
print("Производительность (алгебраический метод):", mflops_algebr, "MFlops")
print("Производительность (cblas):", mflops_cblas, "MFlops")
print("Производительность (оптимизированный метод):", mflops_matmul, "MFlops")

if compare_matrix(C1, C2):
    print("Матрицы C1 и C2 равны.")
else:
    print("Матрицы C1 и C2 не равны.")

if compare_matrix(C1, C3):
    print("Матрицы C1 и C3 равны.")
else:
    print("Матрицы C1 и C3 не равны.")
