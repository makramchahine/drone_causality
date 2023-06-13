## read the points_to_follow.npy file
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

FILE = "points_to_follow.npy"
FREQ = 0.33

def traitement(file:str, freq:int):
    p = np.load(file)
    p0 = [p[0] for i in range(10)]
    p = np.concatenate((p0, p), axis=0)
    print(len(p))
    # plot the points in p as a continuous line with increasing color intensity and big marker size of shape x
    for i in range(len(p)):
        plt.plot(p[i,0], p[i,1], 'o', color=(0,0,i/len(p)), markersize=8, marker='o')

    # filter points in p with a butterworth filter of order 2 and cutoff frequency freq
    N = 1
    for j in range(N):
        f = freq * (j+1)/N
        b, a = butter(6, f, fs=15, btype='low')
        # plot the filtered points in p_filt as a continuous line with increasing (different) color intensity on the same plot with small marker size
        for i in range(len(p)):
            try:
                p_filt = filtfilt(b, a, p[i-50:i,:], axis=0)
            except:
                if i==0:
                    continue
                else:
                    p_filt = p[0:i,:]
            plt.plot(p_filt[-1,0], p_filt[-1,1], 'o', color=(0,i/len(p),0), markersize=2)
    plt.show()

    # # on a 2 fig subplot, plot the x and y coordinates of p and p_filt vs time
    # fig, axs = plt.subplots(2)
    # fig.suptitle('x and y coordinates vs time')
    # axs[0].plot(p[:,0])
    # axs[0].plot(p_filt[:,0])
    # axs[0].set_title('x')
    # axs[1].plot(p[:,1])
    # axs[1].plot(p_filt[:,1])
    # axs[1].set_title('y')
    # for ax in axs.flat:
    #     ax.set(xlabel='time', ylabel='value')
    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()
    # # show plot
    # plt.show()






if __name__ == "__main__":
    traitement(FILE, FREQ)