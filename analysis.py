import pickle
import matplotlib.pyplot as plt

with open('histories/lstm-2020-10-08 18:46:10.953200-history.p', 'rb') as fp:
    history = pickle.load(fp)

print(history)

loss = history['loss']
val_loss = history['val_loss']

plt.plot(loss[:200], label='Training Loss')
plt.plot(val_loss[:200], label='Validation Loss')
plt.title("Cosine Similarity Loss")
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
plt.legend()
plt.show()
