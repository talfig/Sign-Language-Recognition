# mypackage/__init__.py


# Import specific functions from submodules to make them accessible directly
from .datacompressor import compress2npz


def init():
    compress2npz('path_to_dataset', 'compressed_dataset.npz')
