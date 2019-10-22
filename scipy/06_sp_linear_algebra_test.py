# Demonstrates how to use scipy to solve linear algebric equations.
# You need 3 different such equations with the same variables to solve them.
import scipy.linalg as lina

# x + 2y + 3z = 14
# 2x + 5y + z = 15
# 2x + 3y + 8z = 32
coefficients = [[1, 2, 3], [2, 5, 1], [2, 3, 8]]
rhs_values = [14, 15, 32]

solution = lina.solve(coefficients, rhs_values)
print('x =', solution[0])
print('y =', solution[1])
print('z =', solution[2])
