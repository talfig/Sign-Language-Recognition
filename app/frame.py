import cv2

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

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Define smaller rectangle parameters
    # Top-left corner coordinates (shift towards the center)
    top_left = (int(frame_width * 0.35), int(frame_height * 0.3))
    # Bottom-right corner coordinates (shift towards the center)
    bottom_right = (int(frame_width * 0.65), int(frame_height * 0.7))

    # Draw the smaller rectangle on the frame (color: green, thickness: 2)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Display the frame with the smaller rectangle
    cv2.imshow('Hand Detection Frame', frame)

    # Press 'q' to break the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
