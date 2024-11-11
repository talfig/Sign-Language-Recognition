# compressor/data_augmentation.py

import cv2
import numpy as np
import random


# Augmentation functions
def augment_image(image):
    # Apply random augmentation techniques (e.g., flipping, rotation, noise, zoom, shifting, bluring)
    aug_type = random.choice(['flip', 'rotate', 'noise', 'zoom', 'shift', 'blur'])

    if aug_type == 'flip':
        return cv2.flip(image, 1)  # Horizontal flip
    elif aug_type == 'rotate':
        angle = random.choice([90, 180, 270])
        center = (image.shape[1] // 2, image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    elif aug_type == 'zoom':
        zoom_factor = random.uniform(0.7, 1.3)  # Zoom between 70% and 130%
        h, w, _ = image.shape
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        resized_image = cv2.resize(image, (new_w, new_h))

        # If zoomed in (larger), crop the center
        if zoom_factor > 1.0:
            x_start = (new_w - w) // 2
            y_start = (new_h - h) // 2
            return resized_image[y_start:y_start + h, x_start:x_start + w]

        # If zoomed out (smaller), pad the image to maintain size
        else:
            canvas = np.zeros_like(image)
            x_offset = (w - new_w) // 2
            y_offset = (h - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
            return canvas
    elif aug_type == 'shift':
        # Shift by a random number of pixels within a range
        max_shift = 20  # Maximum shift in pixels
        dx, dy = random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    elif aug_type == 'blur':
        # Apply Gaussian blur with a random kernel size
        ksize = random.choice([3, 5, 7])  # Choose a kernel size; larger values produce stronger blurring
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    return image
