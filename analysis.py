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

enviromentNames = list(enviromentNames)
taskNames       = list(taskNames)
modelNames      = list(modelNames)

for enviroment in enviromentNames:
    fig, axes = plt.subplots(len(taskNames), len(modelNames))
    axes = np.reshape(axes, (len(taskNames), len(modelNames)))
    for i, task in enumerate(taskNames):
        for j, model in enumerate(modelNames):
            
            if j == 0:
                axes[i, 0].set_ylabel(task)

            if i == 0:
                axes[0, j].set_title(model)


            try:
                loss = [h["loss"] for h in histories[enviroment][task][model]]
                validationLoss = [h["val_loss"] for h in histories[enviroment][task][model]]
            except KeyError:
                continue

            losses = np.array(loss)
            validations = np.array(validationLoss)

            lossStdDev = np.std(losses, axis=0)
            validationStdDev = np.std(validations, axis=0)

            lossMean = np.mean(losses, axis=0)
            validationMean = np.mean(validations, axis=0)

            print(enviroment, task, model)
            minIdx = np.argmin(validationMean)
            print(f"{validationMean[minIdx]:.3} +/- {validationStdDev[minIdx]:.3}")

            # for l in losses:
            #     axes[i, j].plot(range(len(l)), l, "r", label="losses")
            
            # for v in valdations:
            #     axes[i, j].plot(range(len(v)), v, "b", label="validations")

            axes[i, j].errorbar(range(len(lossMean)), lossMean, yerr=lossStdDev, label="Training Loss")
            axes[i, j].errorbar(range(len(validationMean)), validationMean, yerr=validationStdDev, label="Validation Loss")
            axes[i, j].legend()

            # print(enviroment, task, model)
            # print(loss.shape)
            # print(loss)

            # print(lossStdDev.shape)
            # print(lossStdDev)

            # sys.exit()

        print("")

    fig.suptitle(f"{enviroment} enviroment learning curves")
    plt.show()

