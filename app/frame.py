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
        self.PREDICTION_WINDOW = 10  # Number of frames to average over
        self.PREDICTION_DELAY = 2  # Small delay between predictions (in seconds)
        self.CONFIDENCE_THRESHOLD = 0.7  # Only consider predictions with confidence > 0.7
        self.predictions_queue = deque(maxlen=self.PREDICTION_WINDOW)
        self.last_prediction_time = 0  # Time of the last prediction

        # Set up the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model('../data/weights/asl_crop_v4_1_mobilenet_weights.pth', self.device)
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

        # Add a Delete button with a new color
        self.btn_delete = Button(button_frame, text="Delete", width=12, font=("Arial", 14), bg="#FF4081", fg="white",
                                 command=self.delete_last_sign)
        self.btn_delete.grid(row=1, column=0, padx=10, pady=10)

        # Add a Space button with a new color
        self.btn_space = Button(button_frame, text="Space", width=12, font=("Arial", 14), bg="#00BCD4", fg="white",
                                command=self.add_space)
        self.btn_space.grid(row=1, column=1, padx=10, pady=10)

        # Create a label to display the sentence
        self.sentence = ""

        # Create a frame to serve as the cool outbox for the sentence
        sentence_frame = Frame(self.window, bg="#555555", bd=4, relief=tk.RAISED)
        sentence_frame.pack(pady=20, padx=20)

        # Create a label to display the sentence inside the frame
        self.sentence_label = Label(sentence_frame, text="", font=("Arial", 18), bg="#333333", fg="white",
                                    wraplength=800, padx=10, pady=10)
        self.sentence_label.pack()

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

            # Process the frame
            process_frame(self, frame)

            # Convert the frame to ImageTk format
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.imgtk = ImageTk.PhotoImage(image=img)  # Prevent garbage collection

            # Update canvas with new frame
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)

            # Call update_frame again after a delay (to create a video effect)
            self.window.after(10, self.update_frame)

    def add_to_sentence(self, sign):
        """Add the recognized sign to the sentence and update the label."""
        # Build the sentence by adding the new sign
        self.sentence += sign

        # Update the sentence label
        self.sentence_label.config(text=self.sentence)

    def delete_last_sign(self):
        """Delete the last sign from the sentence."""
        self.sentence = self.sentence[:-1]

        # Update the sentence label
        self.sentence_label.config(text=self.sentence)

    def add_space(self):
        """Add a space to the sentence."""
        self.sentence += " "

        # Update the sentence label
        self.sentence_label.config(text=self.sentence)

    def __del__(self):
        # Release resources
        if self.cap.isOpened():
            self.cap.release()


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = HandDetectionApp(root, "Hand Detection with Mediapipe")
