"""Buffered heatmap class"""
import numpy as np

from collections import deque
from scipy.ndimage.measurements import label


class BufferedHeatmap():
    """BufferedHeatmap"""

    def __init__(self, input_size=(720, 1280), buffer_size=1, threshold=1):
        """Initialize heatmap"""
        self.input_size = input_size
        self.threshold = threshold
        self.heatmaps = deque(maxlen=buffer_size)

    def add_heat(self, bboxes):
        """Create a new local heatmap and add it to the buffer"""
        heatmap = np.zeros(self.input_size).astype(np.float)

        for box in bboxes:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        heatmap[heatmap <= 1] = 0

        self.heatmaps.append(heatmap)

    def get_labels(self):
        """Produce labels based on the combined heatmaps"""
        labelHeatmap = self.get_heatmap()
        return label(labelHeatmap)

    def get_heatmap(self):
        """Return combined heatmap"""
        outputHeatmap = np.zeros(self.input_size).astype(np.float)

        for heatmap in self.heatmaps:
            outputHeatmap += heatmap

        outputHeatmap[outputHeatmap <= 5] = 0

        return outputHeatmap
