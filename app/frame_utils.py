# app/frame_utils.py

import cv2
import mediapipe as mp

# Initialize Mediapipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def calculate_bounding_rectangle(hand_landmarks):
    # Extract x and y coordinates from all the landmarks
    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y for landmark in hand_landmarks.landmark]

    # Calculate the min and max x and y coordinates
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    return min_x, max_x, min_y, max_y


# Function to draw a rectangle using normalized landmark coordinates
def draw_bounding_rectangle(frame, min_x, max_x, min_y, max_y):
    # Get the width and height of the frame (assuming frame.shape contains [height, width, channels])
    h, w, _ = frame.shape

    # Convert normalized coordinates to pixel values based on frame dimensions
    top_left = (int(min_x * w), int(min_y * h))  # Top-left corner
    bottom_right = (int(max_x * w), int(max_y * h))  # Bottom-right corner

    # Draw the rectangle on the frame (color: blue, thickness: 2)
    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)


# Function to draw hand landmarks on the frame
def draw_hand_landmarks(frame, hand_landmarks):
    # Draw landmarks on the frame
    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
