import numpy as np
import time
from multiprocessing import Pool, cpu_count


def generate_random_complex_matrix(size):
    real_part = np.random.rand(size, size).astype(np.float32)
    imag_part = np.random.rand(size, size).astype(np.float32) * 1j
    return (real_part + imag_part).astype(np.complex64)


def multiply_algebr_block(args):
    A, B, block_size, ii, jj, kk = args
    n = A.shape[0]
    result_block = np.zeros((block_size, block_size), dtype=np.complex64)
    for i in range(ii, min(ii + block_size, n)):
        for j in range(jj, min(jj + block_size, n)):
            sum = 0 + 0j
            for k in range(kk, min(kk + block_size, n)):
                sum += A[i, k] * B[k, j]
            result_block[i - ii, j - jj] += sum
    return (ii, jj, result_block)


def multiply_algebr(A, B):
    n = A.shape[0]
    block_size = 32  # размер блока, это можно варьировать
    blocks = [(A, B, block_size, ii, jj, kk) for ii in range(0, n, block_size)
              for jj in range(0, n, block_size)
              for kk in range(0, n, block_size)]
    with Pool(cpu_count()) as p:
        results = p.map(multiply_algebr_block, blocks)

    result = np.zeros((n, n), dtype=np.complex64)
    for (ii, jj, result_block) in results:
        result[ii:ii + block_size, jj:jj + block_size] += result_block
    return result


def multiply_cblas(A, B):
    from scipy.linalg import blas as scipy_blas
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


if __name__ == "__main__":
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
