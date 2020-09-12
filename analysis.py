import pickle
import matplotlib.pyplot as plt

with open('histories/2020-09-11 13:06:59.842404-history.p', 'rb') as fp:
    history = pickle.load(fp)

print(history)

loss = history['loss']
val_loss = history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title("Cosine Similarity Loss")
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
plt.legend()
plt.show()
