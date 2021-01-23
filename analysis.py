import pickle
import matplotlib.pyplot as plt
import numpy as np

# with open('histories/new-ncp-2021_01_14_00_58_35-history.p', 'rb') as fp:
#     history = pickle.load(fp)
# 
# print(history)
# 
# loss = history['loss'][:10]
# val_loss = history['val_loss'][:10]
# 
# plt.plot(loss[:200], label='Training Loss')
# plt.plot(val_loss[:200], label='Validation Loss')
# plt.title("Neighborhood Simple Nav 1: Cosine Similarity Loss")
# plt.xlabel("Epoch Number")
# plt.ylabel("Loss")
# plt.legend()

def plotHistories(ax, histories):
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            if histories[i, j] == '.p':
                continue
            
            print(histories[i, j])
            with open('histories/' + histories[i, j], 'rb') as fp:
                history = pickle.load(fp)
            
            loss = history['loss'][:10]
            val_loss = history['val_loss'][:10]

            ax[i, j].plot(loss, label='Training Loss')
            ax[i, j].plot(val_loss, label='Validation Loss')
            ax[i, j].set_xlabel('Epochs')
            ax[i, j].legend()

redwood_histories = np.array([
    ['.p', '.p', '.p', '.p', '.p', ],
    ['new-ncp-2021_01_14_17_46_58-history.p', 'new-ncp-2021_01_14_18_38_25-history.p', 'new-ncp-2021_01_14_19_27_24-history.p', 'new-ncp-2021_01_14_20_13_55-history.p', 'new-ncp-2021_01_14_21_00_01-history.p', ],
    ['.p', '.p', '.p', '.p', '.p', ],
    ['new-ncp-2021_01_13_16_57_36-history.p', 'new-ncp-2021_01_13_19_04_14-history.p', 'new-ncp-2021_01_13_20_36_40-history.p', 'new-ncp-2021_01_13_23_35_48-history.p', 'new-ncp-2021_01_14_00_17_34-history.p', ],
])


fig, ax = plt.subplots(4, 5)
fig.suptitle('Redwood Forest Tasks')

plotHistories(ax, redwood_histories)

ax[0, 0].set_ylabel('Simple',   labelpad=20, rotation=0)
ax[1, 0].set_ylabel('Chase',    labelpad=20, rotation=0)
ax[2, 0].set_ylabel('Distance', labelpad=20, rotation=0)
ax[3, 0].set_ylabel('Hiking',   labelpad=20, rotation=0)

handles, labels = ax[-1, -1].get_legend_handles_labels()
fig.legend(handles, labels)

plt.show()

neighborhood_histories = np.array([
    ['new-ncp-2021_01_14_00_58_35-history.p', 'new-ncp-2021_01_14_01_54_36-history.p', 'new-ncp-2021_01_14_02_46_07-history.p', 'new-ncp-2021_01_14_11_32_06-history.p', 'new-ncp-2021_01_14_12_08_01-history.p', ],
    ['new-ncp-2021_01_14_13_04_09-history.p', 'new-ncp-2021_01_14_13_48_23-history.p', 'new-ncp-2021_01_14_14_36_45-history.p', 'new-ncp-2021_01_14_15_24_45-history.p', 'new-ncp-2021_01_14_16_14_13-history.p', ],
    ['.p', '.p', '.p', '.p', '.p', ],
    ['.p', '.p', '.p', '.p', '.p', ],
])

fig, ax = plt.subplots(4, 5)
fig.suptitle('Neighborhood Tasks')

plotHistories(ax, neighborhood_histories)

ax[0, 0].set_ylabel('Simple',   labelpad=20, rotation=0)
ax[1, 0].set_ylabel('Chase',    labelpad=20, rotation=0)
ax[2, 0].set_ylabel('Distance', labelpad=20, rotation=0)
ax[3, 0].set_ylabel('Hiking',   labelpad=20, rotation=0)

handles, labels = ax[-1, -1].get_legend_handles_labels()
fig.legend(handles, labels)

plt.show()

# ax[0, 0].set_title('1')
# ax[0, 1].set_title('2')
# ax[0, 2].set_title('3')
# ax[0, 3].set_title('4')
# ax[0, 4].set_title('5')
# ax[0, 0].set_title('1')
# 
# ax[1, 0].set_title('1')
# ax[1, 1].set_title('2')
# ax[1, 2].set_title('3')
# ax[1, 3].set_title('4')
# ax[1, 4].set_title('5')
# 
# ax[2, 0].set_title('1')
# ax[2, 1].set_title('2')
# ax[2, 2].set_title('3')
# ax[2, 3].set_title('4')
# ax[2, 4].set_title('5')
# 
# ax[3, 0].set_title('1')
# ax[3, 1].set_title('2')
# ax[3, 2].set_title('3')
# ax[3, 3].set_title('4')
# ax[3, 4].set_title('5')
# 
# ax[4, 0].set_title('1')
# ax[4, 1].set_title('2')
# ax[4, 2].set_title('3')
# ax[4, 3].set_title('4')
# ax[4, 4].set_title('5')
# 
# ax[5, 0].set_title('1')
# ax[5, 1].set_title('2')
# ax[5, 2].set_title('3')
# ax[5, 3].set_title('4')
# ax[5, 4].set_title('5')
# 
# ax[6, 0].set_title('1')
# ax[6, 1].set_title('2')
# ax[6, 2].set_title('3')
# ax[6, 3].set_title('4')
# ax[6, 4].set_title('5')
# 
# ax[7, 0].set_title('1')
# ax[7, 1].set_title('2')
# ax[7, 2].set_title('3')
# ax[7, 3].set_title('4')
# ax[7, 4].set_title('5')

plt.show()