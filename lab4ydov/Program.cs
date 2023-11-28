import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Задані дані
x = np.array([1, 2, 3, 4, 5])
y = np.array([1.17, 1.51, 3.36, 1.9, 4.8])

# Функція для обчислення поліному Лагранжа
def lagrange_polynomial(x_vals, x_data, y_data):
    result = np.zeros_like(x_vals)
    for i in range(len(x_data)):
        term = y_data[i]
        for j in range(len(x_data)):
            if j != i:
                term *= (x_vals - x_data[j]) / (x_data[i] - x_data[j])
        result += term
    return result

# Значення на проміжку [x1; x5] з кроком h
x1 = np.min(x)
x5 = np.max(x)
h = (x5 - x1) / 200
x_vals = np.arange(x1, x5 + h + h/10, h)  # Доданий епсилон для уникнення числових неточностей

# Значення поліному Лагранжа
y_vals_lagrange = lagrange_polynomial(x_vals, x, y)

# Лінійна інтерполяція
linear_interp = interp1d(x, y, kind='linear', fill_value='extrapolate')
y_vals_linear = linear_interp(x_vals)

# Інтерполяція квадратичним сплайном
quadratic_interp = interp1d(x, y, kind='quadratic', fill_value='extrapolate')
y_vals_quadratic = quadratic_interp(x_vals)

# Інтерполяція кубічним сплайном
cubic_interp = interp1d(x, y, kind='cubic', fill_value='extrapolate')
y_vals_cubic = cubic_interp(x_vals)

# Побудова графіка
plt.plot(x_vals, y_vals_lagrange, label='Lagrange Polynomial', color='red')
plt.plot(x_vals, y_vals_linear, label='Linear Interpolation', linestyle='dashed', color='green')
plt.plot(x_vals, y_vals_quadratic, label='Quadratic Spline', linestyle='dashed', color='blue')
plt.plot(x_vals, y_vals_cubic, label='Cubic Spline', linestyle='dashed', color='purple')
plt.scatter(x, y, color='black', marker='o', label='Data Points')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolation Comparison')
plt.show()
