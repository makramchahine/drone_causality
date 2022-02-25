import pickle
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

SPEED = 0.5
TIMESTEP = 0.05

with open('vectors.p', 'rb') as f:
    vectors = pickle.load(f)

with open('activations.p', 'rb') as f:
    activations = pickle.load(f)

positions = np.zeros((len(vectors) + 1, 3))
for i, v in enumerate(vectors):
    positions[i+1] = v * SPEED * TIMESTEP + positions[i]

positions = positions[1:]

print(positions[0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c = activations)
plt.title('Neuron 0 Activations')
# plt.show()
