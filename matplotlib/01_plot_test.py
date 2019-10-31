import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# Matplotlib objects are composed of two parts, the figure and the axes.
# They can be created together using plt.subplots(figsize)
fig, ax = plt.subplots()
