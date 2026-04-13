import numpy as np
import cv2
from preprocessing import preProcessing
from augmentation import augment


def batch_generator(image_paths, steerings, batch_size, is_training=False):
    num_samples = len(image_paths)

    while True:
        indices = np.arange(num_samples)

        if is_training:
            np.random.shuffle(indices)

        for offset in range(0, num_samples, batch_size):
            batch_indices = indices[offset : offset + batch_size]

            batch_images = []
            batch_steerings = []

            for idx in batch_indices:
                img = cv2.imread(image_paths[idx])
                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                steering = steerings[idx]

                if is_training:
                    img, steering = augment(img, steering)

                img = preProcessing(img)

                batch_images.append(img)
                batch_steerings.append(steering)

            if len(batch_images) == 0:
                continue

            yield np.array(batch_images), np.array(batch_steerings)
