import numpy as np
import matplotlib.pyplot as plt
import time


def z(x, y):
    return x**3 * y**2 * (4 - x - y)


def grad_z(x, y):
    dz_dx = 3 * x**2 * y**2 * (4 - x - y) - x**3 * y**2
    dz_dy = 2 * x**3 * y * (4 - x - y) - x**3 * y**2
    return np.array([dz_dx, dz_dy])


start_point = np.array([2.5, 2.0])
alpha = 0.01  # коэффициент уменьшения шага
epsilon = 1e-10  # критерий остановки по функции
delta = 1e-10  # критерий остановки по аргументу
max_iterations = 1000  # устанавливаю максимальное число итераций, чтобы не словить overflow


def search(start_point, alpha, epsilon, delta, max_iterations):
    point = start_point
    path = [point]
    for iteration in range(max_iterations):
        grad = grad_z(*point)
        next_point = point + alpha * grad
        path.append(next_point)

        if np.linalg.norm(grad) < epsilon:
            print(f"Остановка: малое значение градиента на шаге {iteration + 1}")
            break
        if np.linalg.norm(next_point - point) < delta:
            print(f"Остановка: малое изменение аргумента на шаге {iteration + 1}")
            break

        point = next_point

    return point, path

start_time = time.time()
result, path = search(start_point, alpha, epsilon, delta, max_iterations)
end_time = time.time()

print(f"Найдена точка: ({result[0]}, {result[1]})", )
print("Значение функции в найденной точке:", z(*result))
print("Число итераций:", len(path) - 1)
print("Время работы алгоритма: {:.6f} секунд".format(end_time - start_time))
print()
print("Точка, найденная аналитически: (2, 4/3)")
print("Значение функции в этой точке: ~9.48148148148")


path = np.array(path)
x_path, y_path = path[:, 0], path[:, 1]
z_path = np.array([z(x, y) for x, y in path])


x = np.linspace(0, 2.5, 200)
y = np.linspace(0, 2, 200)
X, Y = np.meshgrid(x, y)
Z = z(X, Y)

# 3d график
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.plot(x_path, y_path, z_path, color='r', marker='o', label="Траектория")
ax.scatter(result[0], result[1], z(*result), color='blue', s=100, label="Локальный максимум")
ax.set_title("Траектория на графике функции")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()

# проекция
ax2 = fig.add_subplot(122)
ax2.contourf(X, Y, Z, cmap='viridis', levels=100)
ax2.plot(x_path, y_path, color='r', marker='o', label="Траектория")
ax2.scatter(result[0], result[1], color='blue', s=100, label="Локальный максимум", zorder=5)
ax2.set_title("Проекция на плоскость")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend()

plt.tight_layout()
plt.show()
