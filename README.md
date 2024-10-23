<h1 align="center"> Sign-Language-Recognition </h1>

<p align="center">
    <a href="https://github.com/talfig/Assembler">
      <img src="https://github.com/talfig/Sign-Language-Recognition/blob/main/hand_landmarks.png">
    </a>
</p>

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
