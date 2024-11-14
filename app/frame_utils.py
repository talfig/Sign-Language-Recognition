# app/frame_utils.py

import cv2
import time
import mediapipe as mp
from utils import *

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define the color mapping for different landmarks in BGR format
colors = {
    'red': (45, 46, 255),       # Red in BGR format for landmarks 0, 1, 5, 9, 13, 17
    'orange': (184, 231, 255),  # Light orange in BGR format for landmarks 2, 3, 4
    'purple': (130, 61, 134),   # Purple for landmarks 6, 7, 8
    'yellow': (6, 206, 255),    # Yellow for landmarks 10, 11, 12
    'green': (46, 255, 49),     # Green for landmarks 14, 15, 16
    'blue': (190, 103, 23),     # Blue in BGR format for landmarks 18, 19, 20
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
    font_scale = 0.75
    thickness = 1
    text_size = cv2.getTextSize(prediction_sign, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = text_top_left[0] + (text_bottom_right[0] - text_top_left[0] - text_size[0]) // 2
    text_y = text_top_left[1] + (text_box_height + text_size[1]) // 2

    # Display the prediction text inside the filled box (color: white)
    if prediction_sign is not None:
        cv2.putText(frame, prediction_sign, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


def detect_hand_bounds(frame, hand_landmarks, prediction_sign):
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


def smooth_predictions(app):
    """
    Smooth predictions by returning the most frequent prediction within a set window size.
    """
    if len(app.predictions_queue) == app.PREDICTION_WINDOW:
        # Return the most common prediction
        smoothed_prediction = max(set(app.predictions_queue), key=app.predictions_queue.count)

        # Clear the queue after processing the prediction
        app.predictions_queue.clear()

        return smoothed_prediction

    return None


def process_frame(app, frame):
    # Make a prediction based on the current frame
    prediction = make_prediction(app, frame)

    # If a prediction is made, check the delay for adding it to the sentence
    if prediction:
        current_time = time.time()
        # Check if the delay period has passed
        if current_time - app.last_prediction_time > app.PREDICTION_DELAY:
            # Update the last prediction time and add the prediction to the sentence
            app.last_prediction_time = current_time
            app.add_to_sentence(prediction)


def make_prediction(app, frame):
    results = app.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx > 1:
                print(f"Warning: Too many hands detected")
                break

            orig_features_mask = np.zeros_like(frame)
            draw_hand_features(frame, hand_landmarks)
            extract_hand_features_mask(orig_features_mask, hand_landmarks)
            mirror_features_mask = cv2.flip(orig_features_mask, 1)

            orig_input_image = app.transform(orig_features_mask).unsqueeze(0).to(app.device)
            mirror_input_image = app.transform(mirror_features_mask).unsqueeze(0).to(app.device)

            with torch.no_grad():
                orig_output = app.model(orig_input_image)
                mirror_output = app.model(mirror_input_image)

                final_output = torch.max(orig_output, mirror_output)
                confidence, predicted_class = torch.max(final_output, 1)

                if confidence.item() > app.CONFIDENCE_THRESHOLD:
                    predicted_sign = LabelMapper.index_to_label(predicted_class.item())

                    # Add the prediction to the queue and smooth it
                    app.predictions_queue.append(predicted_sign)
                    smoothed_prediction = smooth_predictions(app)

                    if smoothed_prediction:
                        # Draw detected hand bounds with the smoothed prediction
                        detect_hand_bounds(frame, hand_landmarks, smoothed_prediction)

                        # Display the smoothed prediction immediately (on frame)
                        return smoothed_prediction  # Return prediction to be displayed immediately

    # Return None if no valid prediction is made
    return None
