# Demonstrates the mathemathical functions in numpy.
import numpy as np

# Trigonomic functions require an input of angles.
a = np.array(range(0, 361, 30))
print(a)

print('Sine of different angles:')
# Convert to radians by multiplying with pi/180
print(np.sin(a*np.pi/180))
print()

print('Cosine values for angles in array:')
print(np.cos(a*np.pi/180))
print()

print('Tangent values for given angles:')
print(np.tan(a*np.pi/180))
