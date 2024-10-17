# compressor/__init__.py

from compressor.data_compressor import *

# Call the init function when the module is run as a script
if __name__ == "__main__":
    compress_to_npz('../ASL-crop', '../data/compressed_asl_crop.npz')
