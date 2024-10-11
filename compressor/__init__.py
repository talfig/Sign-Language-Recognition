# mypackage/__init__.py


# Import specific functions from submodules to make them accessible directly
from compressor.data_compressor import *

# Call the init function when the module is run as a script
if __name__ == "__main__":
    compress2npz('C:/Users/xbpow/Downloads/Sign-Language-Recognition/ASL-crop', '../data/compressed_asl_crop.npz')
