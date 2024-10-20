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


# Function to draw a bounding box and a filled box for the prediction text
def draw_bounding_box(frame, min_x, max_x, min_y, max_y, prediction_sign=""):
    # Get the width and height of the frame (assuming frame.shape contains [height, width, channels])
    h, w, _ = frame.shape

    # Convert normalized coordinates to pixel values based on frame dimensions
    top_left = (int(min_x * w), int(min_y * h))  # Top-left corner
    bottom_right = (int(max_x * w), int(max_y * h))  # Bottom-right corner

    # Adjust the top of the bounding box to leave space for text
    padding = 20  # Padding above the bounding box for the filled text box (reduced padding)
    text_box_height = 20  # Height of the filled text box (smaller height)
    top_left_with_padding = (top_left[0], max(0, top_left[1] - padding))

    # Draw the main bounding box (color: blue, thickness: 2)
    cv2.rectangle(frame, top_left_with_padding, bottom_right, (255, 0, 0), 2)

    # Draw the filled rectangle for the text (color: blue)
    text_top_left = (top_left_with_padding[0], top_left_with_padding[1] - text_box_height)
    text_bottom_right = (bottom_right[0], top_left_with_padding[1])
    cv2.rectangle(frame, text_top_left, text_bottom_right, (255, 0, 0), cv2.FILLED)

    # Calculate the position for centering the text inside the filled rectangle
    font_scale = 0.75  # Smaller font scale
    thickness = 1  # Thicker text to make it bolder
    text_size = cv2.getTextSize(prediction_sign, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = text_top_left[0] + (text_bottom_right[0] - text_top_left[0] - text_size[0]) // 2
    text_y = text_top_left[1] + (text_box_height + text_size[1]) // 2

    # Display the prediction text inside the filled box (color: white)
    if prediction_sign is not None:
        cv2.putText(frame, prediction_sign, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


def detect_hand_landmarks(frame, hand_landmarks, prediction_sign):
    # Calculate the bounding rectangle based on hand landmarks
    min_x, max_x, min_y, max_y = get_hand_bounding_box(hand_landmarks)

    # Draw the bounding rectangle directly on the frame
    draw_bounding_box(frame, min_x, max_x, min_y, max_y, prediction_sign)


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


def extract_hand_features_mask(black_background, hand_landmarks):
    draw_palm_connections(black_background, hand_landmarks)
    draw_hand_landmarks(black_background, hand_landmarks)


# Function to draw landmarks with red circles and connections with green lines
def draw_hand_features(frame, hand_landmarks):
    # Set colors: red for landmarks, green for connections
    red_color = (0, 0, 255)   # Red color for the circles (landmarks)
    green_color = (0, 255, 0)  # Green color for the lines (connections)

    # Draw hand landmarks and connections with specified colors
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=red_color, thickness=-1, circle_radius=4),  # Filled red circles
        mp_drawing.DrawingSpec(color=green_color, thickness=2)  # Green lines (connections)
    )


def display_predictions(frame, hand_landmarks, confidence, predicted_sign, predictions_queue, confidence_threshold, prediction_window):
    if predicted_sign is not None:
        # Ensure the prediction has high confidence
        if confidence.item() > confidence_threshold:
            # Append the prediction to the queue
            predictions_queue.append(predicted_sign)

            # Get the smoothed prediction
            smoothed_prediction = smooth_predictions(predictions_queue, prediction_window)

            if smoothed_prediction is not None:
                # Call the function to detect hand landmarks
                detect_hand_landmarks(frame, hand_landmarks, smoothed_prediction)


def process_frame(frame, hands, model, transform, device, multi_predictions_queue, confidence_threshold, prediction_window):
    # Process the frame to detect hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Create a black background frame
            hand_features_frame = np.zeros_like(frame)

            # Draw palm connections and landmarks
            draw_hand_features(frame, hand_landmarks)

            # Extract hand features from the frame based on detected landmarks
            extract_hand_features_mask(hand_features_frame, hand_landmarks)

            # Preprocess the frame with landmarks for model prediction
            input_image = transform(hand_features_frame).unsqueeze(0).to(device)

            # Make predictions using the model
            with torch.no_grad():
                output = model(input_image)
                confidence, predicted_class = torch.max(output, 1)  # Get the index of the highest prediction score

                # Convert the predicted class index to a sign label (string)
                predicted_sign = LabelMapper.index_to_label(predicted_class.item())

                # Display the predicted sign
                display_predictions(frame, hand_landmarks, confidence, predicted_sign, multi_predictions_queue[idx], confidence_threshold, prediction_window)


def smooth_predictions(predictions_queue, prediction_window):
    if len(predictions_queue) == prediction_window:
        # Calculate the mode (most frequent prediction in the window)
        most_common_prediction = max(set(predictions_queue), key=predictions_queue.count)
        return most_common_prediction
    else:
        return None  # Not enough frames for smoothing yet
