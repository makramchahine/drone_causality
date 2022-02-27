import cv2
import numpy as np
from numpy import ndarray

TEXT_BOX_HEIGHT = 30
ARROW_BOX_HEIGHT = 100 + TEXT_BOX_HEIGHT
VEL_MAX = 2
YAW_MAX = 2
ARROW_COLOR = (0, 255, 0)
ARROW_THICKNESS = 4


def show_vel_cmd(vel_cmd: ndarray, img_width: int):
    """
    Draws arrows corresponding to the values in vel_cmd. The arrows correspond to the directions on a mode 2 rc
    controller. The left arrow vertical is throttle and horizontal is yaw. The right arrow vertical is pitch and the
    horizontal is roll

    :param vel_cmd: ndarray of shape 1x4 with commands pitch, roll, throt, yaw
    :param img_width: width of image to be returned
    :return: Image with both vel_cmd text and arrows drawn on it
    """
    vel_text = show_vel_text(vel_cmd, img_width)
    vel_cmd = np.squeeze(vel_cmd, axis=0)
    arrow_img = np.zeros((ARROW_BOX_HEIGHT-TEXT_BOX_HEIGHT, img_width, 3), dtype=np.uint8)
    left_stick_x = img_width // 3
    right_stick_x = img_width * 2 // 3
    arrow_y = ARROW_BOX_HEIGHT // 2
    max_arrow_len = img_width // 3
    arrow_scale = max_arrow_len / VEL_MAX

    yaw_img = cv2.arrowedLine(arrow_img, (left_stick_x, arrow_y),
                              (left_stick_x + int(vel_cmd[3] * arrow_scale), arrow_y), ARROW_COLOR, ARROW_THICKNESS)
    throt_img = cv2.arrowedLine(yaw_img, (left_stick_x, arrow_y),
                                (left_stick_x, arrow_y + int(-vel_cmd[2] * arrow_scale)), ARROW_COLOR, ARROW_THICKNESS)
    roll_img = cv2.arrowedLine(throt_img, (right_stick_x, arrow_y),
                               (right_stick_x + int(vel_cmd[1] * arrow_scale), arrow_y), ARROW_COLOR, ARROW_THICKNESS)
    pitch_img = cv2.arrowedLine(roll_img, (right_stick_x, arrow_y),
                                (right_stick_x, arrow_y + int(-vel_cmd[0] * arrow_scale)), ARROW_COLOR, ARROW_THICKNESS)

    return np.concatenate([vel_text, pitch_img], axis=0)


def show_vel_text(vel_cmd: ndarray, img_width: int):
    """
    Draws vel_cmd as text on an image for visualization
    :param vel_cmd: ndarray of shape 1xn that has velocity to display
    :param img_width: width of image that should be returned
    :return: Image with vel_cmd drawn on as text
    """
    text_img = np.zeros((TEXT_BOX_HEIGHT, img_width, 3), dtype=np.uint8)
    vel_rounded = str([round(vel, 2) for vel in vel_cmd[0]])
    cv2.putText(text_img, vel_rounded, (0, TEXT_BOX_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1,
                cv2.LINE_AA)
    return text_img
