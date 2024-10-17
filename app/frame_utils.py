# app/frame_utils.py

import cv2
import mediapipe as mp

# Initialize Mediapipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Function to draw a larger rectangle on the frame
def draw_bounding_rectangle(frame, frame_width, frame_height):
    # Adjust the coordinates to make the rectangle larger
    top_left = (int(frame_width * 0.3), int(frame_height * 0.25))
    bottom_right = (int(frame_width * 0.7), int(frame_height * 0.75))

    # Draw the larger rectangle on the frame (color: blue, thickness: 2)
    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)


# Function to draw hand landmarks on the frame
def draw_hand_landmarks(frame, landmarks):
    # Draw landmarks on the frame
    for hand_landmarks in landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
