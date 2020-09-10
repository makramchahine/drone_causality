import pickle
import matplotlib.pyplot as plt

with open('histories/history.p', 'rb') as fp:
    history = pickle.load(fp)

loss = history['loss']

plt.plot(loss)
plt.title("Mean Squared Training Loss")
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
plt.show()

