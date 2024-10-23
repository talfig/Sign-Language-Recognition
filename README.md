<h1 align="center"> Sign-Language-Recognition </h1>

<p align="center">
    <a href="https://github.com/talfig/Assembler">
      <img src="https://github.com/talfig/Sign-Language-Recognition/blob/main/hand_landmarks.png">
    </a>
</p>

## Project Overview

The core objective of this project is to provide accurate and fast classification of hand signs based on real-time input. It is implemented using several well-known libraries:

1. **MediaPipe Hands Model**: This model detects and tracks hand landmarks in real-time, offering high accuracy in hand position recognition. It provides 21 key points (landmarks) for each hand and tracks multiple hands in a video frame, which are used for further sign classification.
   
2. **PyTorch Classification Model**: Initially, a **ResNet18** model was tested but resulted in lower accuracy. Consequently, a pre-trained **MobileNet** architecture was adopted, which provides superior accuracy in classifying hand signs. The model has been trained using a dataset of hand sign images, with its weights stored in the file `asl_crop_v2_mobilenet_weights_epoch_10.pth`. This model outputs probabilities for different hand sign classes, and a confidence threshold is applied to ensure accurate predictions.

## Main Features

1. **Hand Tracking and Detection**: The **MediaPipe Hands** model is responsible for detecting the user's hand in the video feed and extracting key hand landmarks. These landmarks are essential for determining hand orientation and positioning, which feeds into the classification process.
   
2. **Sign Classification**: Once the hand is detected and the landmarks are identified, the processed landmarks are passed into the PyTorch classification model. The classifier predicts the hand sign based on the detected landmarks. The model has been trained on a variety of hand gestures to recognize different signs accurately.

3. **Real-time Video Processing**: The application continuously processes webcam frames, applying the hand tracking model, running sign classification, and displaying the results in a GUI. A **confidence threshold** of 0.7 is applied, meaning that only predictions with high certainty are shown to the user. Additionally, predictions are averaged over the last 5 frames to smooth out any jitter or instability in the real-time predictions.

4. **Custom Hand Landmarks Display**: The hand landmarks are visually represented in the output video stream. Each part of the hand (fingers, palm, etc.) is color-coded for better clarity. This visualization helps the user see the points being tracked and how they correspond to the predicted hand sign.

## Application Flow

1. **Webcam Input**: The app captures live video input from the webcam.
2. **Hand Detection**: The **MediaPipe** model processes each frame, detects the hands, and extracts landmarks.
3. **Hand Sign Classification**: The detected landmarks are passed to the PyTorch model, which predicts the hand sign.
4. **Display**: The video feed is displayed through a graphical interface, with hand landmarks and classification results overlaid.

## Models Used

### 1. **MediaPipe Hands**:
   - MediaPipe Hands provides robust hand detection and tracking capabilities by identifying 21 landmarks on each hand. It works well under different lighting conditions and can detect multiple hands in a single frame.
   - This model ensures that only the precise hand region is analyzed, which is essential for the classification step.

### 2. **PyTorch Hand Sign Classifier**:
   - The classification model is based on the **MobileNet** architecture, which has been fine-tuned for the task of hand sign recognition. MobileNet is a lightweight model designed for mobile and embedded vision tasks, making it an efficient choice for real-time applications.
   - The classifier takes hand landmarks as input and predicts the hand sign from a predefined set of classes. The model used in this project was trained for 10 epochs, and its weights are stored in the file `asl_crop_v2_mobilenet_weights_epoch_10.pth`.

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/talfig/sign-language-recognition.git
   cd sign-language-recognition
   ```
2. **Install dependencies**:

   Ensure Python 3.x is installed. Then, install the required libraries by running:
   
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the Application**:

   To launch the app and start the webcam-based hand sign detection:
   
   ```bash
   python app/frame.py
   ```

## CUDA and cuDNN Installation Guide for Project (Windows & Ubuntu)

### 1. Downloading and Installing CUDA

#### For Windows:
1. Visit the official NVIDIA CUDA Toolkit website: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
2. Select **Windows** as your operating system and download the appropriate version (CUDA 11.2 is suggested for compatibility with TensorFlow and PyTorch).
3. Run the downloaded installer and follow the installation instructions. Choose **Express Install** or **Custom Install** depending on your preference.
4. After installation, verify the CUDA installation by running the following command in **Command Prompt**:

    ```bash
    nvcc --version
    ```

#### For Ubuntu:
1. Visit the official NVIDIA CUDA Toolkit website: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
2. Select **Linux** as your operating system and download the appropriate version.
3. Follow these commands to install CUDA:

    ```bash
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
    sudo apt-get update
    sudo apt-get -y install cuda
    ```

4. Verify the installation with:

    ```bash
    nvcc --version
    ```

---

### 2. Downloading and Installing cuDNN

#### For Windows:
1. Visit the NVIDIA cuDNN library page: [cuDNN Download](https://developer.nvidia.com/cudnn).
2. Download the cuDNN version compatible with your CUDA installation.
3. Unzip the cuDNN package and copy the files into the appropriate CUDA directories (usually located in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`):
   - Copy the contents of the `bin` folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`.
   - Copy the contents of the `include` folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include`.
   - Copy the contents of the `lib` folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64`.

#### For Ubuntu:
1. Visit the NVIDIA cuDNN library page: [cuDNN Download](https://developer.nvidia.com/cudnn).
2. Download the cuDNN version compatible with your CUDA installation.
3. Install cuDNN by running:

    ```bash
    tar -xzvf cudnn-linux-x86_64-8.x.x.x_cuda11.2-archive.tar.xz
    sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
    ```

---

### 3. Adding CUDA and cuDNN to Environment Variables

#### For Windows:
1. Open **Control Panel** > **System and Security** > **System**.
2. Click **Advanced system settings** on the left, then click **Environment Variables**.
3. Under **System variables**, find `Path`, select it, and click **Edit**.
4. Add the following to the list of paths (adjust the CUDA version if needed):
    - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
    - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp`
5. Click **OK** to save the changes.

#### For Ubuntu:
1. Open your `.bashrc` or `.zshrc` file (depending on your shell):

    ```bash
    nano ~/.bashrc
    ```

2. Add the following lines at the end of the file:

    ```bash
    export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    ```

3. Save the file and run:

    ```bash
    source ~/.bashrc
    ```

# Troubleshooting PyTorch Not Detecting GPU

If you're encountering an error indicating that your PyTorch installation does not support CUDA, follow these steps to resolve the issue:

### Steps to Fix the CUDA Error in PyTorch

#### 1. Check PyTorch Installation

Ensure that you have installed a version of PyTorch that supports CUDA. Run the following commands in your Python environment:

```bash
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

#### 2. Reinstall PyTorch with CUDA Support

If CUDA is not available, you may need to reinstall PyTorch with the correct CUDA version. Use the following command, ensuring it matches your installed CUDA version:

For example, if you have CUDA 12.6 installed:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126
```

You can find the correct command on the official PyTorch installation page.

#### 3. Verify CUDA Installation

Make sure your CUDA installation is correctly set up and that your environment variables point to the correct directories. Check that CUDA_PATH and CUDA_PATH_V11_2 are set correctly.

#### 4. Check NVIDIA Driver

Ensure that your NVIDIA driver is compatible with the version of CUDA you are using. You can check the installed driver version using:

```bash
nvidia-smi
```

#### 5. Test CUDA Availability

After reinstalling PyTorch with CUDA support, run the following command again to check if CUDA is now available:

```bash
import torch
print(torch.cuda.is_available())
```
