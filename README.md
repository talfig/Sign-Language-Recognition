# Sign-Language-Recognition

(https://github.com/talfig/Sign-Language-Recognition/blob/main/ASL.jpg)

## CUDA and cuDNN Installation Guide for Project

### 1. Downloading and Installing CUDA

To install CUDA, follow these steps:

1. Visit the official NVIDIA CUDA Toolkit website: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
2. Select your operating system and download the appropriate version (we recommend using CUDA 11.2 for compatibility with TensorFlow and PyTorch).
3. Follow the installation instructions for your system, or use the commands below for Ubuntu:

```bash
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo apt-get update
sudo apt-get -y install cuda
```

### 2. Downloading and Installing cuDNN

1. Visit the NVIDIA cuDNN library page: [cuDNN Download](https://developer.nvidia.com/cudnn).
2. Download the cuDNN version compatible with your CUDA installation.
3. Follow the installation steps based on your system. For Linux, use:

```bash
tar -xzvf cudnn-linux-x86_64-8.x.x.x_cuda11.2-archive.tar.xz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### 3. Adding CUDA and cuDNN to Environment Variables

To make CUDA and cuDNN available globally, add them to your environment variables:

Open your .bashrc or .zshrc file (depending on your shell):

```bash
nano ~/.bashrc
```

Add the following lines at the end of the file:

```bash
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Save the file and run:

```bash
source ~/.bashrc
```

## What To Do When TensorFlow is Not Detecting Your GPU

If TensorFlow is not detecting your GPU, it can slow down deep learning processes. Below are some common reasons and solutions to fix the issue.

### Why is TensorFlow not Detecting Your GPU?

1. **Missing Dependencies:** TensorFlow needs specific dependencies to run on a GPU, like the CUDA toolkit and cuDNN library.
2. **Outdated/Incompatible Drivers:** Your GPU drivers might be outdated or incompatible with TensorFlow.
3. **Incorrect Installation:** TensorFlow might not be installed with GPU support.
4. **Missing Environment Variables:** TensorFlow may not locate CUDA due to missing environment variables.
5. **Hardware Limitations:** Older GPUs might not meet the minimum requirements.

### How to Fix TensorFlow Not Detecting Your GPU

#### 1. Install Required Dependencies

Make sure you have the latest NVIDIA GPU drivers, CUDA Toolkit, and cuDNN library:

* **NVIDIA Drivers:** Download from NVIDIA Drivers.
* **CUDA Toolkit:** Download from CUDA Toolkit.
* **cuDNN Library:** Download from cuDNN.

#### 2. Update GPU Drivers

* **NVIDIA:** Download from NVIDIA Drivers.
* **AMD:** Download from AMD Drivers.
* **Intel:** Download from Intel Drivers.

#### 3. Reinstall TensorFlow with GPU Support

**Using pip:**

```bash
pip uninstall tensorflow
pip install tensorflow-gpu
```

**Using Anaconda:**

```bash
conda uninstall tensorflow
conda install tensorflow-gpu
```

#### 4. Set Environment Variables

For `Windows`:

```bash
setx PATH "%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\extras\CUPTI\lib64"
setx PATH "%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin"
```

For `Linux`:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
```

#### 5. Check Your Hardware

Ensure your GPU meets the minimum requirements for TensorFlow:

* NVIDIA® GPU with CUDA®** Compute Capability 3.5 or higher
* CUDA® Toolkit 11.0 or higher
* cuDNN 8.0 or higher
* 8 GB of RAM or more

## Help for PyTorch: Fixing CUDA Error

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
