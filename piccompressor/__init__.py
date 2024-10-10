# mypackage/__init__.py


# Import specific functions from submodules to make them accessible directly
from .compression import preprocess_and_save


def init():
    preprocess_and_save('path_to_dataset', 'compressed_dataset.npz', size=(200, 200))
