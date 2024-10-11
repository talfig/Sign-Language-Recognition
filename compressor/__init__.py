# mypackage/__init__.py


# Import specific functions from submodules to make them accessible directly
from compressor.data_compressor import *
from compressor.data_augmentation import *


def init():
    compress2npz('C:/Users/xbpow/Downloads/Sign-Language-Recognition/ASL', 'compressed_asl.npz')


# Call the init function when the module is run as a script
if __name__ == "__main__":
    init()
