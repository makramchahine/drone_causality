import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import argparse
import sys
import numpy as np
import pandas as pd

historyFile = 'histories/ncp_encoder_decoder-2021_06_24_23_38_00-history-rev=13.0.p'

with open(historyFile, 'rb') as fp:
    history = pickle.load(fp)

plt.plot(history["loss"], label='Training')
plt.plot(history["val_loss"], label='Validation')
plt.title('Demonstration Task State Vector Prediction')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()