import argparse
import os
from pathlib import Path

import cv2
import pandas as pd

DISPLAY_SIZE_INCREASE = 3


def slice_sequence(data_path: str, output_path: str):
    Path(output_path).mkdir(exist_ok=True, parents=True)
    flight_folders = sorted(os.listdir(data_path))
    print(flight_folders)

    for flight in flight_folders:
        from_flight = [flight in dir for dir in os.listdir(output_path)]
        if any(from_flight):
            print(f"Flight {flight} already detected in {sum(from_flight)} output sliced runs. Skipping")
            continue

        index = 0
        seq = [[], []]
        io = 0

        while True:
            try:
                im_num = str(index).zfill(6)  # format index with zeros to the left
                img = cv2.imread(os.path.join(data_path, flight, f"{im_num}.png"))  # load frame
                height, width, channels = img.shape
                upscaled_img = cv2.resize(img, (DISPLAY_SIZE_INCREASE * width, DISPLAY_SIZE_INCREASE * height))
                cv2.imshow('img', upscaled_img)

                k = cv2.waitKey(0)
                if k == 100:  # d key for next image
                    index += 1
                elif k == 97:  # a key for previous image
                    index = max([0, index - 1])
                elif k == 32:  # space bar to delimit sequence
                    seq[io].append(index)
                    io = 1 - io
                    index += 1
                    print(seq)
                elif k == 27:  # escape to exit
                    break
                else:
                    pass
            except:
                print('Flight end at frame number ' + str(index - 1))
                if len(seq[0]) > len(seq[1]):
                    print('Please end open sequence')
                    index -= 1
                else:
                    break

        ####################################################################################################################
        ### Create a folder per sequence with the corresponding images and sliced csv labels ###
        ####################################################################################################################

        n_seq = len(seq[0])
        data = pd.read_csv(os.path.join(data_path, flight, 'data_out.csv'))

        for i in range(0, n_seq):
            seq_start = seq[0][i]
            seq_end = seq[1][i]
            fname = flight + '_' + str(i + 1)
            seq_out_dir = os.path.join(output_path, fname)
            Path(seq_out_dir).mkdir(parents=True, exist_ok=True)
            df = data[seq_start:seq_end]
            df.to_csv(os.path.join(output_path, fname, 'data_out.csv'), index=False)
            # need to loop to renumber images
            for j in range(seq_start, seq_end):
                os.system(
                    f"cp {os.path.join(data_path, flight, str(j).zfill(6))}.png {os.path.join(seq_out_dir, str(j - seq_start).zfill(6))}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    slice_sequence(args.data_path, args.output_path)
