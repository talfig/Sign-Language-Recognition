# app/frame.py

from torchvision import transforms
from collections import deque
from app.frame_utils import *
from utils import *

# Initialize the Mediapipe Hands model
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the width and height of the frame (optional, depends on your webcam)
frame_width = 640
frame_height = 480
cap.set(3, frame_width)
cap.set(4, frame_height)

# Define constants
PREDICTION_WINDOW = 5  # Number of frames to average over
CONFIDENCE_THRESHOLD = 0.8  # Only consider predictions with confidence > 0.7

# Store the predictions in a queue for smoothing
predictions_queue = deque(maxlen=PREDICTION_WINDOW)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model('../data/mobilenet_weights_epoch_10.pth', len(string.ascii_uppercase), device)
model.eval()  # Set the model to evaluation mode

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to RGB since Mediapipe works with RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # Create a mirrored version of the frame
    mirrored_frame = cv2.flip(frame, 1)  # Flip horizontally to create the mirror effect

    # Process both the original and mirrored frames
    predicted_sign_orig, output_orig = process_frame(frame, hands, model, transform, device)
    predicted_sign_mirror, output_mirror = process_frame(mirrored_frame, hands, model, transform, device)

    if output_orig is not None and output_mirror is not None:
        # Take the maximum of the two outputs
        final_output = torch.max(output_orig, output_mirror)
        confidence, final_prediction = torch.max(final_output, 1)

        # Ensure the prediction has high confidence
        if confidence.item() > CONFIDENCE_THRESHOLD:
            # Append the prediction to the queue
            predictions_queue.append(final_prediction.item())

            # Get the smoothed prediction
            smoothed_prediction = get_smoothed_prediction(predictions_queue, PREDICTION_WINDOW)

            if smoothed_prediction is not None:
                # Get the predicted label for the smoothed prediction
                predicted_sign = LabelMapper.index_to_label(smoothed_prediction)

                # Display the predicted sign on the frame
                cv2.putText(frame, f"Predicted Sign: {predicted_sign}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2, cv2.LINE_AA)

    # If hand landmarks are detected, draw them on the mask
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate the bounding rectangle based on hand landmarks
            min_x, max_x, min_y, max_y = calculate_bounding_rectangle(hand_landmarks)

            # Draw the bounding rectangle directly on the frame
            draw_bounding_rectangle(frame, min_x, max_x, min_y, max_y)

    # Display the frame with the landmarks
    cv2.imshow('Hand Detection with Mediapipe', frame)

    # Press 'q' to break the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
