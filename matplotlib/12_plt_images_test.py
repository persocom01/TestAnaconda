# Demonstrates using matplotlib to display images.
import matplotlib.pyplot as plt

with open('./datasets/Innocence.jpg', 'rb') as f:
    img = plt.imread(f)

fig, ax = plt.subplots()
ax.imshow(img)
ax.axis('off')
