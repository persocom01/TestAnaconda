# Demonstrates the mathemathical functions in numpy.
import numpy as np

# 360 degrees = 2pi radians.
# np.arcsin(sin)
sin = np.sin(np.pi / 2)
print('sin:', sin)
print('arcsin:', np.arcsin(sin))
cos = np.cos(np.pi)
print('cos:', cos)
print('arccos:', np.arccos(cos))
tan = np.tan(2 * np.pi / 360 * 45)
print('tan:', np.tan(2 * np.pi / 360 * 45))
print('arctan:', np.arctan(tan))
print('tan in degrees:', np.degrees(tan))

n = 5.55
# np.around(n, dec_places) rounds up decimals 0.5 and above to 1.
print('around:', np.around(n, 1))
# np.floor(n) rounds down to the nearest int.
print('floor:', np.floor(n))
# Opposite of floor.
print('ceiling:', np.ceil(n))
