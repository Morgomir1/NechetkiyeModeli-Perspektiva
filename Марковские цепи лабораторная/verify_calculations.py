import numpy as np

# Задание 1 - проверка
P1 = np.array([
    [0.7, 0.1, 0.1, 0.1],
    [0.2, 0.6, 0.0, 0.2],
    [0.2, 0.0, 0.5, 0.3],
    [0.0, 0.0, 0.0, 1.0]
])
P0_1 = np.array([1.0, 0.0, 0.0, 0.0])

print("Задание 1:")
P1_k1 = P0_1 @ P1
P1_k2 = P1_k1 @ P1
P1_k3 = P1_k2 @ P1
print(f"k=1: {P1_k1}")
print(f"k=2: {P1_k2}")
print(f"k=3: {P1_k3}")
print()

# Задание 2 - проверка
P2 = np.array([
    [0.5, 0.4, 0.1, 0.0],
    [0.3, 0.4, 0.3, 0.0],
    [0.0, 0.3, 0.5, 0.2],
    [0.0, 0.0, 0.4, 0.6]
])
P0_2 = np.array([0.0, 0.0, 1.0, 0.0])

print("Задание 2:")
P2_k3 = P0_2 @ np.linalg.matrix_power(P2, 3)
print(f"P(3) = {P2_k3}")
print()

# Задание 3 - проверка
P3 = np.array([
    [0.5, 0.5],
    [0.4, 0.6]
])
R = np.array([
    [9, 3],
    [3, -7]
])
P0_3 = np.array([0.1, 0.9])

def calc_income(P, P_trans, R):
    income = 0.0
    for i in range(len(P)):
        for j in range(len(P)):
            income += P[i] * P_trans[i, j] * R[i, j]
    return income

print("Задание 3:")
P3_k0 = P0_3.copy()
G0 = calc_income(P3_k0, P3, R)
print(f"k=0: P={P3_k0}, G={G0:.2f}")

P3_k1 = P3_k0 @ P3
G1 = calc_income(P3_k1, P3, R)
print(f"k=1: P={P3_k1}, G={G1:.2f}")

P3_k2 = P3_k1 @ P3
G2 = calc_income(P3_k2, P3, R)
print(f"k=2: P={P3_k2}, G={G2:.2f}")

P3_k3 = P3_k2 @ P3
G3 = calc_income(P3_k3, P3, R)
print(f"k=3: P={P3_k3}, G={G3:.2f}")


