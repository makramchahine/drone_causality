import argparse
import os

import cv2

DISPLAY_SIZE_INCREASE = 2


def select_targets_all_runs(session_dir: str):
    flight_folders = sorted(os.listdir(session_dir))
    all_targets = []
    for flight in flight_folders:
        all_targets.extend(select_target_locations(os.path.join(session_dir, flight)))

    return all_targets


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
            print(f"Detected click. Press a or d to save location")
            # displaying the coordinates
            # on the Shell
            click_loc = (x // DISPLAY_SIZE_INCREASE, y // DISPLAY_SIZE_INCREASE)

    index = 0
    seq_targets = []
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
                seq_targets.append((img_path, click_loc))
                click_loc = None
        except IndexError:
            print('Flight end at frame number ' + str(index - 1))
            break
    return seq_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    print(select_target_locations(args.run_dir))
