# app/frame_utils.py

import cv2
import mediapipe as mp
from utils import *

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


def process_frame(frame, hands, model, transform, device):
    """
    Process a given frame to detect hand landmarks, draw bounding rectangles, and make predictions using the model.

    Args:
        frame: The frame to be processed.
        hands: The Mediapipe hands detection object.
        model: The trained PyTorch model for prediction.
        transform: The torchvision transform to preprocess the frame.
        device: The device (e.g., 'cuda' or 'cpu') for model inference.

    Returns:
        final_prediction: The predicted sign for the frame.
        output: The model's output for further comparison.
    """
    # Convert the frame to RGB since Mediapipe works with RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            draw_hand_landmarks(frame, hand_landmarks)

        # Preprocess the frame with landmarks for model prediction
        input_image = transform(frame).unsqueeze(0).to(device)

        # Make predictions using the model
        with torch.no_grad():
            output = model(input_image)
            _, predicted_class = torch.max(output, 1)  # Get the index of the highest prediction score

        predicted_sign = LabelMapper.index_to_label(predicted_class.item())

        return predicted_sign, output

    return None, None


def get_smoothed_prediction(predictions_queue, prediction_window):
    if len(predictions_queue) == prediction_window:
        # Calculate the mode (most frequent prediction in the window)
        most_common_prediction = max(set(predictions_queue), key=predictions_queue.count)
        return most_common_prediction
    else:
        return None  # Not enough frames for smoothing yet
