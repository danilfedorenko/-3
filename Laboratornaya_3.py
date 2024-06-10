import timeit
from collections import defaultdict, deque


def start():
    # Задаем данные
    routes = [
        [1, 2, 3, 4],
        [2, 5, 6],
        [1, 7, 8, 6],
        [1, 8, 5],
        [1, 2, 4, 6],
    ]
    source = 1
    destination = 6

    # Реализация через матрицу
    print("Реализация через матрицу:")
    evaluate_time(lambda: matrix_approach(routes, source, destination))

    # Реализация через связный список
    print("\nРеализация через связный список:")
    evaluate_time(lambda: linked_list_approach(routes, source, destination))

    # Реализация через встроенные структуры
    print("\nРеализация через встроенные структуры:")
    evaluate_time(lambda: built_in_approach(routes, source, destination))

    # Реализация через очередь
    print("\nРеализация через очередь:")
    evaluate_time(lambda: bfs_approach(routes, source, destination))

    # Реализация через очередь без использования граф и вершин
    print("\nРеализация через очередь без использования граф и вершин:")
    evaluate_time(lambda: basic_queue(routes, source, destination))


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

    depth_first_search_matrix(graph, source, destination, current_path, all_paths)

    # Вывод найденных путей
    print("Найденные пути (через матрицу):")
    for path in all_paths:
        print(" ".join(map(str, path)))

    # Поиск наилучшего пути по условиям задачи
    optimal_path = find_best_path(all_paths, routes)

    # Вывод лучшего пути и номера автобуса
    best_bus_number = find_best_bus(optimal_path, routes)
    print("Кратчайший путь:")
    print(" ".join(map(str, optimal_path)))
    print(f"\nАвтобус с кратчайшим путем: {best_bus_number}")


def linked_list_approach(routes, source, destination):
    graph = defaultdict(list)
    all_paths = []
    current_path = []

    # Заполнение графа
    for route in routes:
        for i in range(len(route) - 1):
            graph[route[i]].append(route[i + 1])
            graph[route[i + 1]].append(route[i])

    depth_first_search_linked_list(graph, source, destination, current_path, all_paths)

    # Вывод найденных путей
    print("Найденные пути (через связный список):")
    for path in all_paths:
        print(" ".join(map(str, path)))

    # Поиск наилучшего пути по условиям задачи
    optimal_path = find_best_path(all_paths, routes)

    # Вывод лучшего пути и номера автобуса
    best_bus_number = find_best_bus(optimal_path, routes)
    print("Кратчайший путь:")
    print(" ".join(map(str, optimal_path)))
    print(f"\nАвтобус с кратчайшим путем: {best_bus_number}")


def depth_first_search_linked_list(graph, current, destination, current_path, all_paths):
    current_path.append(current)

    if current == destination:
        all_paths.append(current_path.copy())
    else:
        for neighbor in graph[current]:
            if neighbor not in current_path:
                depth_first_search_linked_list(graph, neighbor, destination, current_path, all_paths)

    current_path.pop()


def built_in_approach(routes, source, destination):
    graph = defaultdict(set)
    all_paths = []
    current_path = []

    # Заполнение графа
    for route in routes:
        for i in range(len(route) - 1):
            graph[route[i]].add(route[i + 1])
            graph[route[i + 1]].add(route[i])

    depth_first_search_built_in(graph, source, destination, current_path, all_paths)

    # Вывод найденных путей
    print("Найденные пути (через встроенные структуры):")
    for path in all_paths:
        print(" ".join(map(str, path)))

    # Поиск наилучшего пути по условиям задачи
    optimal_path = find_best_path(all_paths, routes)

    # Вывод лучшего пути и номера автобуса
    best_bus_number = find_best_bus(optimal_path, routes)
    print("Кратчайший путь:")
    print(" ".join(map(str, optimal_path)))
    print(f"\nАвтобус с кратчайшим путем: {best_bus_number}")


def depth_first_search_built_in(graph, current, destination, current_path, all_paths):
    current_path.append(current)

    if current == destination:
        all_paths.append(current_path.copy())
    else:
        for neighbor in graph[current]:
            if neighbor not in current_path:
                depth_first_search_built_in(graph, neighbor, destination, current_path, all_paths)

    current_path.pop()


def bfs_aproach(routes, source, destination):
    graph = defaultdict(list)
    all_paths = []

    # Заполнение графа
    for route in routes:
        for i in range(len(route) - 1):
            graph[route[i]].append(route[i + 1])
            graph[route[i + 1]].append(route[i])

    queue = deque([(source, [source])])

    while queue:
        current, path = queue.popleft()
        if current == destination:
            all_paths.append(path)
        else:
            for neighbor in graph[current]:
                if neighbor not in path:
                    queue.append((neighbor, path + [neighbor]))

    # Вывод найденных путей
    print("Найденные пути (через очередь):")
    for path in all_paths:
        print(" ".join(map(str, path)))

    # Поиск наилучшего пути по условиям задачи
    optimal_path = find_best_path(all_paths, routes)

    # Вывод лучшего пути и номера автобуса
    best_bus_number = find_best_bus(optimal_path, routes)
    print("Кратчайший путь:")
    print(" ".join(map(str, optimal_path)))
    print(f"\nАвтобус с кратчайшим путем: {best_bus_number}")


def basic_queue(routes, source, destination):
    all_paths = []
    queue = deque([(source, [source])])

    while queue:
        current, path = queue.popleft()
        if current == destination:
            all_paths.append(path)
        else:
            for route in routes:
                if current in route:
                    current_index = route.index(current)
                    # Check neighbors in the route
                    if current_index > 0 and route[current_index - 1] not in path:
                        queue.append((route[current_index - 1], path + [route[current_index - 1]]))
                    if current_index < len(route) - 1 and route[current_index + 1] not in path:
                        queue.append((route[current_index + 1], path + [route[current_index + 1]]))

    # Вывод найденных путей
    print("Найденные пути (через очередь без использования граф и вершин):")
    for path in all_paths:
        print(" ".join(map(str, path)))

    # Поиск наилучшего пути по условиям задачи
    optimal_path = find_best_path(all_paths, routes)

    # Вывод лучшего пути и номера автобуса
    best_bus_number = find_best_bus(optimal_path, routes)
    print("Кратчайший путь:")
    print(" ".join(map(str, optimal_path)))
    print(f"\nАвтобус с кратчайшим путем: {best_bus_number}")


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
