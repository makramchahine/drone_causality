import pickle
import matplotlib.pyplot as plt

with open('training-histories/history.p', 'rb') as fp:
    history = pickle.load(fp)

loss = history['loss']

plt.plot(loss)
plt.show()

lossManual = [49.5115, 29.6346, 17.7142, 

