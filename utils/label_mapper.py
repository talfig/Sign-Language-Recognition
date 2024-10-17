# utils/label_mapper.py

import string


class LabelMapper:
    def __init__(self):
        """
        Initializes the LabelMapper by creating a label-to-index map and an index-to-label map.
        """
        # Create a label set '0-9' and 'A-Z'
        labels = list(string.digits) + list(string.ascii_uppercase)

        # Create a label-to-index map
        self.label_to_index_map = {label: idx for idx, label in enumerate(labels)}

        # Create an index-to-label map
        self.index_to_label_map = {idx: label for label, idx in self.label_to_index_map.items()}

    def label_to_index(self, label):
        """
        Convert a label to its corresponding index.
        Args:
            label: The label to convert.
        Returns:
            The index corresponding to the label.
        """
        return self.label_to_index_map.get(label, None)

    def index_to_label(self, index):
        """
        Convert an index to its corresponding label.
        Args:
            index: The index to convert.
        Returns:
            The label corresponding to the index.
        """
        return self.index_to_label_map.get(index, None)
