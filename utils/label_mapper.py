# utils/label_mapper.py

import string


class LabelMapper:
    # Create a label list containing digits (0-9) and uppercase letters (A-Z)
    labels = list(string.digits) + list(string.ascii_uppercase)

    # Create a label-to-index map
    label_to_index_map = {label: idx for idx, label in enumerate(labels)}

    # Create an index-to-label map
    index_to_label_map = {idx: label for idx, label in enumerate(labels)}

    @staticmethod
    def label_to_index(label):
        """
        Convert a label to its corresponding index.
        Args:
            label: The label to convert.
        Returns:
            The index corresponding to the label, or None if the label is not found.
        """
        return LabelMapper.label_to_index_map.get(label, None)

    @staticmethod
    def index_to_label(index):
        """
        Convert an index to its corresponding label.
        Args:
            index: The index to convert.
        Returns:
            The label corresponding to the index, or None if the index is not found.
        """
        return LabelMapper.index_to_label_map.get(index, None)