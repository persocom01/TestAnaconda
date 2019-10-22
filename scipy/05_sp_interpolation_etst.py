# Demonstrates the various tools in scipy that help you connect the dots
# in a graph.
import numpy as np
import scipy.interpolate as inter
import matplotlib.pyplot as plt

x = np.linspace(0, 4, 12)
y = np.cos(x**2/3+4)
f = inter.interp1d(x, y, kind='linear')
f2 = inter.interp1d(x, y, kind='cubic')
xnew = np.linspace(0, 4, 30)

plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic', 'nearest'], loc='best')
plt.show()
