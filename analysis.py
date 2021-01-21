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


    for historyFile in os.listdir(args.history_dir + '/' + domain ):

        # map each model type to its training histories
        model = historyFile[:historyFile.index('-')]
        if model not in histories[enviroment][task]:
            histories[enviroment][task][model] = []
            modelNames.add(model)

        with open(args.history_dir + '/' + domain + '/' + historyFile, 'rb') as fp:
            histories[enviroment][task][model].append(pickle.load(fp))

        print(enviroment, task)

for enviroment in histories:
    print(len(taskNames), len(modelNames))
    fig, axes = plt.subplots(len(taskNames), len(modelNames))
    axes = np.reshape(axes, (len(taskNames), len(modelNames)))
    for i, task in enumerate(histories[enviroment]):
        for j, model in enumerate(histories[enviroment][task]):
            print(i, j)
            loss = [h["loss"] for h in histories[enviroment][task][model]]
            valdationLoss = [h["val_loss"] for h in histories[enviroment][task][model]]

            losses = np.array(loss)
            valdations = np.array(valdationLoss)

            lossStdDev = np.std(losses, axis=0)
            validationStdDev = np.std(valdations, axis=0)

            lossMean = np.mean(losses, axis=0)
            valdationMean = np.mean(valdations, axis=0)

            # for l in losses:
            #     axes[i, j].plot(range(len(l)), l, "r", label="losses")
            
            # for v in valdations:
            #     axes[i, j].plot(range(len(v)), v, "b", label="validations")

            axes[i, j].errorbar(range(len(lossMean)), lossMean, yerr=lossStdDev, label="Training Loss")
            axes[i, j].errorbar(range(len(valdationMean)), valdationMean, yerr=validationStdDev, label="Validation Loss")
            axes[i, j].legend()
            
            if j == 0:
                axes[i, j].set_ylabel(task)

            if i == 1:
                axes[0, j].set_title(model)

            # print(enviroment, task, model)
            # print(loss.shape)
            # print(loss)

            # print(lossStdDev.shape)
            # print(lossStdDev)

            # sys.exit()

    fig.suptitle(f"{enviroment} enviroment learning curves")
    plt.show()

