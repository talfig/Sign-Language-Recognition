# app/frame_utils.py

import cv2
import mediapipe as mp
from utils import *

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define the color mapping for different landmarks in BGR format
colors = {
    'red': (0, 0, 255),         # Red in BGR format for landmarks 0, 1, 5, 9, 13, 17
    'orange': (215, 242, 255),  # Light orange in BGR format for landmarks 2, 3, 4
    'purple': (128, 0, 128),    # Purple for landmarks 6, 7, 8
    'yellow': (14, 208, 255),   # Yellow for landmarks 10, 11, 12
    'green': (0, 255, 0),       # Green for landmarks 14, 15, 16
    'blue': (255, 0, 0),        # Blue in BGR format for landmarks 18, 19, 20
    'palm': (128, 128, 128)     # Gray for palm connections
}

# Lists to store the indices of the different landmark groups
red_indices = [0, 1, 5, 9, 13, 17]
body_color_indices = [2, 3, 4]
purple_indices = [6, 7, 8]
yellow_indices = [10, 11, 12]
green_indices = [14, 15, 16]
blue_indices = [18, 19, 20]

# Define palm connections (gray lines)
palm_connections = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (17, 0)]


def get_hand_bounding_box(hand_landmarks):
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
def draw_bounding_box(frame, min_x, max_x, min_y, max_y):
    # Get the width and height of the frame (assuming frame.shape contains [height, width, channels])
    h, w, _ = frame.shape

    # Convert normalized coordinates to pixel values based on frame dimensions
    top_left = (int(min_x * w), int(min_y * h))  # Top-left corner
    bottom_right = (int(max_x * w), int(max_y * h))  # Bottom-right corner

    # Draw the rectangle on the frame (color: blue, thickness: 2)
    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)


def detect_hand_landmarks(frame, hands):
    # Process the frame to detect hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # If hand landmarks are detected, draw them on the mask
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate the bounding rectangle based on hand landmarks
            min_x, max_x, min_y, max_y = get_hand_bounding_box(hand_landmarks)

            # Draw the bounding rectangle directly on the frame
            draw_bounding_box(frame, min_x, max_x, min_y, max_y)


# Function to draw palm connections
def draw_palm_connections(frame, hand_landmarks):
    for connection in palm_connections:
        start_idx, end_idx = connection
        start_point = hand_landmarks.landmark[start_idx]
        end_point = hand_landmarks.landmark[end_idx]
        start_coords = (int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0]))
        end_coords = (int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0]))
        cv2.line(frame, start_coords, end_coords, colors['palm'], 2)


# Function to draw landmarks with custom colors
def draw_hand_landmarks(frame, hand_landmarks):
    # Draw finger connections using the default MediaPipe method
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )

    # Draw landmarks with specified colors
    for i, landmark in enumerate(hand_landmarks.landmark):
        coords = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))

        # Color landmarks based on their group
        if i in red_indices:
            cv2.circle(frame, coords, 5, colors['red'], -1)
        elif i in body_color_indices:
            cv2.circle(frame, coords, 5, colors['orange'], -1)
        elif i in purple_indices:
            cv2.circle(frame, coords, 5, colors['purple'], -1)
        elif i in yellow_indices:
            cv2.circle(frame, coords, 5, colors['yellow'], -1)
        elif i in green_indices:
            cv2.circle(frame, coords, 5, colors['green'], -1)
        elif i in blue_indices:
            cv2.circle(frame, coords, 5, colors['blue'], -1)


# Function to process the frame and draw hand landmarks
def draw_hand_features(frame, hand_landmarks):
    # Draw palm connections and landmarks on the frame
    draw_palm_connections(frame, hand_landmarks)
    draw_hand_landmarks(frame, hand_landmarks)


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
    # Process the frame to detect hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Create a black background frame
    hand_features_frame = np.zeros_like(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw palm connections and landmarks
            draw_hand_features(frame, hand_landmarks)

            # Extract hand features from the frame based on detected landmarks
            draw_hand_features(hand_features_frame, hand_landmarks)

        # Preprocess the frame with landmarks for model prediction
        input_image = transform(hand_features_frame).unsqueeze(0).to(device)

        # Make predictions using the model
        with torch.no_grad():
            output = model(input_image)
            confidence, predicted_class = torch.max(output, 1)  # Get the index of the highest prediction score

        # Convert the predicted class index to a sign label (string)
        predicted_sign = LabelMapper.index_to_label(predicted_class.item())

        # Return the confidence and the predicted sign (string)
        return confidence, predicted_sign

    # If no input image is provided or frame couldn't be processed
    return None, None


def smooth_predictions(predictions_queue, prediction_window):
    if len(predictions_queue) == prediction_window:
        # Calculate the mode (most frequent prediction in the window)
        most_common_prediction = max(set(predictions_queue), key=predictions_queue.count)
        return most_common_prediction
    else:
        return None  # Not enough frames for smoothing yet
