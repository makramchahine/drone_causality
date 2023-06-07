import argparse
import json
import os

import cv2
from tqdm import tqdm

DISPLAY_SIZE_INCREASE = 4


def select_targets_all_runs(session_dir: str):
    flight_folders = sorted(os.listdir(session_dir))
    all_targets_png = []
    all_targets_dir = []
    for flight in tqdm(flight_folders):
        # if "_" not in flight: # skip non sliced sequences
        #     continue
        png_targets, dir_targets = select_target_locations(os.path.join(session_dir, flight))
        all_targets_png.extend(png_targets)
        all_targets_dir.extend(dir_targets)

    return all_targets_png, all_targets_dir


def select_target_locations(run_dir: str):
    """
    Convenience script that records image path and click location and allows you to use a and d to look through an
    image sequence
    """
    click_loc = None

    def click_event(event, x, y, flags, params):
        nonlocal click_loc
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # print(f"Detected click. Press a or d to save location")
            # displaying the coordinates
            # on the Shell
            click_loc = (x // DISPLAY_SIZE_INCREASE, y // DISPLAY_SIZE_INCREASE)

    index = 0
    png_targets = []
    dir_targets = []
    imgs = sorted(os.listdir(run_dir))

    while True:
        try:
            img_path = os.path.join(run_dir, imgs[index])
            img = cv2.imread(img_path)  # load frame
            height, width, channels = img.shape
            upscaled_img = cv2.resize(img, (DISPLAY_SIZE_INCREASE * width, DISPLAY_SIZE_INCREASE * height))
            cv2.imshow('img', upscaled_img)
            cv2.setMouseCallback("img", click_event)
            k = cv2.waitKey(0)
            if k == 100:  # d key for next image
                index += 1
            elif k == 97:  # a key for previous image
                index = max([0, index - 1])
            elif k == 27:  # escape to exit
                break
            else:
                pass
            if click_loc is not None:
                img_dir = os.path.dirname(img_path)
                png_targets.append([img_path, click_loc])
                dir_targets.append([img_dir, click_loc])
                click_loc = None
        except IndexError:
            print('Flight end at frame number ' + str(index - 1))
            break
    return png_targets, dir_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    png_targets, dir_targets = select_target_locations(args.run_dir)
    print(png_targets)
    print(dir_targets)
    with open("output_png.json", "w") as f:
        json.dump(png_targets, f)

    with open("output_dir.json", "w") as f:
        json.dump(dir_targets, f)
