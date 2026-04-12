import cv2
import numpy as np
from random import random


def flip(img, steering):
    return cv2.flip(img, 1), -steering


def brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    factor = 0.4 + np.random.uniform()   
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def zoom(img):
    h, w = img.shape[:2]
    scale = np.random.uniform(1.0, 1.3)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    top  = (new_h - h) // 2
    left = (new_w - w) // 2
    return resized[top:top + h, left:left + w]


def pan(img, steering):
    h, w = img.shape[:2]
    max_shift_x = w * 0.1
    max_shift_y = h * 0.1
    dx = np.random.uniform(-max_shift_x, max_shift_x)
    dy = np.random.uniform(-max_shift_y, max_shift_y)
    M  = np.float32([[1, 0, dx], [0, 1, dy]])
    steering_adjustment = dx / max_shift_x * 0.1
    return cv2.warpAffine(img, M, (w, h)), steering + steering_adjustment


def rotate(img):
    h, w = img.shape[:2]
    angle = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def augment(img, steering):
    if random() < 0.5:
        img, steering = flip(img, steering)
    if random() < 0.5:
        img = brightness(img)
    if random() < 0.5:
        img = zoom(img)
    if random() < 0.5:
        img, steering = pan(img, steering)
    if random() < 0.5:
        img = rotate(img)
    return img, steering
