import pickle
import matplotlib.pyplot as plt

with open('histories/2020-09-11 04:19:10.567995-history.p', 'rb') as fp:
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
