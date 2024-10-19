# app/frame.py

import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
from torchvision import transforms
from collections import deque
from app.frame_utils import *
from utils import *


class HandDetectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("900x650")
        self.window.configure(bg="#333333")  # Dark background for modern look

        # Initialize the Mediapipe Hands model
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

        # Define constants
        self.PREDICTION_WINDOW = 5  # Number of frames to average over
        self.CONFIDENCE_THRESHOLD = 0.7  # Only consider predictions with confidence > 0.7
        self.predictions_queue = deque(maxlen=self.PREDICTION_WINDOW)

        num_classes = len(string.digits) + len(string.ascii_uppercase)

        # Set up the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model('../data/asl_crop_mobilenet_weights_epoch_10.pth', num_classes, self.device)
        self.model.eval()

        # Set up image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize the webcam feed
        self.cap = cv2.VideoCapture(0)

        # Check if the webcam is opened correctly
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        # Create a title label
        title_label = Label(self.window, text="Hand Sign Detection", font=("Arial", 24, "bold"), bg="#333333",
                            fg="#FFFFFF")
        title_label.pack(pady=10)

        # Create a frame for the video display
        video_frame = Frame(self.window, bg="#333333")
        video_frame.pack(pady=10)

        # Create a canvas for video display
        self.canvas = tk.Canvas(video_frame, width=640, height=480, bg="#222222", bd=2, relief=tk.SUNKEN)
        self.canvas.pack()

        # Create a frame for the buttons
        button_frame = Frame(self.window, bg="#333333")
        button_frame.pack(pady=20)

        # Add start/stop buttons with enhanced style
        self.btn_start = Button(button_frame, text="Start", width=12, font=("Arial", 14), bg="#4CAF50", fg="white",
                                command=self.start_video)
        self.btn_start.grid(row=0, column=0, padx=10)

        self.btn_stop = Button(button_frame, text="Stop", width=12, font=("Arial", 14), bg="#f44336", fg="white",
                               command=self.stop_video)
        self.btn_stop.grid(row=0, column=1, padx=10)

        # Add an Exit button
        self.btn_exit = Button(button_frame, text="Exit", width=12, font=("Arial", 14), bg="#555555", fg="white",
                               command=self.window.quit)
        self.btn_exit.grid(row=0, column=2, padx=10)

        # Create a label to display the predicted sign
        self.predicted_label = Label(self.window, text="Predicted Sign: ", font=("Arial", 20), bg="#333333",
                                     fg="#FFFFFF")
        self.predicted_label.pack(pady=10)

        self.running = False  # Control the flow of video feed
        self.imgtk = None  # Store the image to prevent garbage collection

        self.window.mainloop()

    def start_video(self):
        """Start video capture."""
        if not self.running:
            self.running = True
            self.update_frame()

    def stop_video(self):
        """Stop video capture."""
        self.running = False

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()

            if not ret:
                print("Error: Could not read frame.")
                pass

            # Call the function to detect hand landmarks
            detect_hand_landmarks(frame, self.hands)

            # Process the frame
            confidence, predicted_sign = process_frame(frame, self.hands, self.model, self.transform, self.device)

            if predicted_sign is not None:
                # Ensure the prediction has high confidence
                if confidence.item() > self.CONFIDENCE_THRESHOLD:
                    # Append the prediction to the queue
                    self.predictions_queue.append(predicted_sign)

                    # Get the smoothed prediction
                    smoothed_prediction = smooth_predictions(self.predictions_queue, self.PREDICTION_WINDOW)

                    if smoothed_prediction is not None:
                        self.predicted_label.config(text=f"Predicted Sign: {smoothed_prediction}")

            else:
                # Display the predicted sign on the frame
                self.predicted_label.config(text="No hands detected")

            # Convert the frame to ImageTk format
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.imgtk = ImageTk.PhotoImage(image=img)  # Prevent garbage collection

            # Update canvas with new frame
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)

            # Call update_frame again after a delay (to create a video effect)
            self.window.after(10, self.update_frame)

    def __del__(self):
        # Release resources
        if self.cap.isOpened():
            self.cap.release()


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = HandDetectionApp(root, "Hand Detection with Mediapipe")
