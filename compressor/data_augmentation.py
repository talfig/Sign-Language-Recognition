# compressor/data_augmentation.py

import cv2
import numpy as np
import random


# Augmentation functions
def augment_image(image):
    # Apply random augmentation techniques (e.g., flipping, rotation)
    aug_type = random.choice(['flip', 'rotate', 'noise'])

    if aug_type == 'flip':
        return cv2.flip(image, 1)  # Horizontal flip
    elif aug_type == 'rotate':
        angle = random.choice([90, 180, 270])
        center = (image.shape[1] // 2, image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    elif aug_type == 'noise':
        noise = np.random.randint(0, 50, image.shape, dtype='uint8')
        return cv2.add(image, noise)
    return image
