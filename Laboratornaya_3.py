import timeit
from collections import defaultdict

def start():
    # Ввод данных
    count_routes = int(input("Введите количество маршрутов: "))
    routes = []

    for i in range(count_routes):
        route = list(map(int, input(f"Введите остановки для маршрута {i + 1}, разделенные пробелом: ").split()))
        routes.append(route)

    source, destination = map(int, input("Введите начальную и конечную остановки, разделенные пробелом: ").split())

    # Реализация через матрицу
    print("Реализация через матрицу:")
    evaluate_time(lambda: matrix_approach(routes, source, destination))

    # Реализация через связный список
    print("\nРеализация через связный список:")
    evaluate_time(lambda: linked_list_approach(routes, source, destination))

    # Реализация через встроенные структуры
    print("\nРеализация через встроенные структуры:")
    evaluate_time(lambda: built_in_approach(routes, source, destination))


def evaluate_time(func):
    elapsed_time = timeit.timeit(func, number=1)
    print(f"Время выполнения: {elapsed_time:.6f} секунд")


def depth_first_search_matrix(graph, current, destination, current_path, all_paths):
    current_path.append(current)

    if current == destination:
        all_paths.append(current_path.copy())
    else:
        for neighbor in range(len(graph)):
            if graph[current][neighbor] == 1 and neighbor not in current_path:
                depth_first_search_matrix(graph, neighbor, destination, current_path, all_paths)

    current_path.pop()


def matrix_approach(routes, source, destination):
    n = max(max(route) for route in routes)
    graph = [[0] * (n + 1) for _ in range(n + 1)]
    all_paths = []
    current_path = []

    # Заполнение графа
    for route in routes:
        for i in range(len(route) - 1):
            graph[route[i]][route[i + 1]] = 1
            graph[route[i + 1]][route[i]] = 1

    # Поиск всех путей между парой вершин
    depth_first_search_matrix(graph, source, destination, current_path, all_paths)

    # Вывод найденных путей
    print("Найденные пути (через матрицу):")
    for path in all_paths:
        print(" ".join(map(str, path)))

    # Поиск наилучшего пути по условиям задачи
    optimal_path = find_best_path(all_paths, routes)

    # Вывод лучшего пути и номера автобуса
    best_bus_number = find_best_bus(optimal_path, routes)
    print("Лучший путь:")
    print(" ".join(map(str, optimal_path)))
    print(f"\nНомер автобуса: {best_bus_number}")


def depth_first_search_list(graph, current, destination, current_path, all_paths):
    current_path.append(current)

    if current == destination:
        all_paths.append(current_path.copy())
    else:
        for neighbor in graph[current]:
            if neighbor not in current_path:
                depth_first_search_list(graph, neighbor, destination, current_path, all_paths)

    current_path.pop()


def linked_list_approach(routes, source, destination):
    graph = defaultdict(list)
    all_paths = []
    current_path = []

    # Заполнение графа
    for route in routes:
        for i in range(len(route) - 1):
            graph[route[i]].append(route[i + 1])
            graph[route[i + 1]].append(route[i])

    # Поиск всех путей между парой вершин
    depth_first_search_list(graph, source, destination, current_path, all_paths)

    # Вывод найденных путей
    print("Найденные пути (через связный список):")
    for path in all_paths:
        print(" ".join(map(str, path)))

    # Поиск наилучшего пути по условиям задачи
    optimal_path = find_best_path(all_paths, routes)

    # Вывод лучшего пути и номера автобуса
    best_bus_number = find_best_bus(optimal_path, routes)
    print("Лучший путь:")
    print(" ".join(map(str, optimal_path)))
    print(f"\nНомер автобуса: {best_bus_number}")


def depth_first_search_built_in(graph, current, destination, current_path, all_paths):
    current_path.append(current)

    if current == destination:
        all_paths.append(current_path.copy())
    else:
        for neighbor in graph[current]:
            if neighbor not in current_path:
                depth_first_search_built_in(graph, neighbor, destination, current_path, all_paths)

    current_path.pop()


def built_in_approach(routes, source, destination):
    graph = defaultdict(list)
    all_paths = []
    current_path = []

    # Заполнение графа
    for route in routes:
        for i in range(len(route) - 1):
            graph[route[i]].append(route[i + 1])
            graph[route[i + 1]].append(route[i])

    # Поиск всех путей между парой вершин
    depth_first_search_built_in(graph, source, destination, current_path, all_paths)

    # Вывод найденных путей
    print("Найденные пути (через встроенные структуры):")
    for path in all_paths:
        print(" ".join(map(str, path)))

    # Поиск наилучшего пути по условиям задачи
    optimal_path = find_best_path(all_paths, routes)

    # Вывод лучшего пути и номера автобуса
    best_bus_number = find_best_bus(optimal_path, routes)
    print("Лучший путь:")
    print(" ".join(map(str, optimal_path)))
    print(f"\nНомер автобуса: {best_bus_number}")


def find_best_path(all_paths, routes):
    min_weight = float('inf')
    best_path = None

    for path in all_paths:
        weight = compute_path_weight(path, routes)
        if weight < min_weight:
            min_weight = weight
            best_path = path

    return best_path


def compute_path_weight(path, routes):
    weight = 0
    transfers = 0

    for i in range(len(path) - 1):
        weight += 1
        if not is_direct_bus(path[i], path[i + 1], routes):
            transfers += 1

    weight += transfers * 3
    return weight


def is_direct_bus(stop1, stop2, routes):
    for route in routes:
        found_first = False
        for stop in route:
            if stop == stop1:
                found_first = True
            if found_first and stop == stop2:
                return True
    return False


def find_best_bus(path, routes):
    bus_usage = {}

    for i in range(len(path) - 1):
        for j, route in enumerate(routes):
            if path[i] in route and path[i + 1] in route:
                if j + 1 not in bus_usage:
                    bus_usage[j + 1] = 0
                bus_usage[j + 1] += 1

    return max(bus_usage, key=bus_usage.get)


if __name__ == "__main__":
    start()