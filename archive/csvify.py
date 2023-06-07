import pickle
import os
import matplotlib.pyplot as plt
import argparse
import sys

import numpy as np

parser = argparse.ArgumentParser(description='Plot training history of deepdrone project')
parser.add_argument('--history_dir', type=str, default="history", help='The path to the training histories')
args = parser.parse_args()

enviromentNames = set()
taskNames       = set()
modelNames      = set()

histories = dict()
for domain in os.listdir(args.history_dir):

    task = domain[:domain.index('-')]
    enviroment = domain[domain.index('-')+1:]

    if enviroment not in histories:
        histories[enviroment] = dict()
        enviromentNames.add(enviroment)

    if task not in histories[enviroment]:
        histories[enviroment][task] = dict()
        taskNames.add(task)


    for i, historyFile in enumerate(os.listdir(args.history_dir + '/' + domain)):

        # map each model type to its training histories
        model = historyFile[:historyFile.index('-')]
        if model not in histories[enviroment][task]:
            histories[enviroment][task][model] = []
            modelNames.add(model)

        with open(args.history_dir + '/' + domain + '/' + historyFile, 'rb') as fp:
            history = pickle.load(fp)
            histories[enviroment][task][model].append(history)

            np.savetxt(f"csv/{enviroment}-{task}-{model}-training-loss-{len(histories[enviroment][task][model])}", history["loss"], delimiter=",")
            np.savetxt(f"csv/{enviroment}-{task}-{model}-validation-loss-{len(histories[enviroment][task][model])}", history["val_loss"], delimiter=",")
