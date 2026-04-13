import cv2
import numpy as np
from random import random


def flip(img, steering):
    img = cv2.flip(img, 1)
    steering = -steering
    return img, steering


def brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    factor = 0.5 + np.random.uniform()
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def pan(img, steering):
    h, w = img.shape[:2]
    max_shift_x = w * 0.2
    dx = np.random.uniform(-max_shift_x, max_shift_x)
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (w, h))

    # recover back to the center
    steering += (dx / max_shift_x) * 0.4
    return img, steering


def rotate(img, steering):
    h, w = img.shape[:2]
    angle = np.random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))
    steering += angle * 0.002
    return img, steering


def zoom(img):
    h, w = img.shape[:2]
    scale = np.random.uniform(1.0, 1.15)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    top = (new_h - h) // 2
    left = (new_w - w) // 2
    return resized[top : top + h, left : left + w]


def augment(img, steering):
    """
     Applies random augmentations (flip, pan,brightness,zoom ,rotate) to the input image and adjusts the steering according to it.
    """

    if random() < 0.5:
        img, steering = flip(img, steering)

    if random() < 0.9:
        img, steering = pan(img, steering)

    if random() < 0.3:
        img = brightness(img)

    if random() < 0.1:
        img = zoom(img)
        
    if random() < 0.1:
        img, steering = rotate(img, steering)

    return img, steering
