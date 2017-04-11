import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
import collections
import matplotlib.cm as cm

class HeatMapper():
    def  __init__(self, img_shape):
        try:
            if type(img_shape) != tuple:
                raise TypeError
            elif (img_shape[0] == 0) or (img_shape[1] == 0):
                raise TypeError
        except Exception as e:
            print('img_shape argument is not a valid tuple, enter valid image shape')

        self.img_shape = img_shape
        self.heatmap = np.zeros(self.img_shape).astype(np.float)
        self.thresholded_heatmap = np.zeros(self.img_shape).astype(np.float)
        self.multi_boxes_list = collections.deque(maxlen=10)
        self.i = 0

    def add_heat(self, bbox_list):
        self.heatmap = np.zeros(self.img_shape).astype(np.float)

        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return self.heatmap

    def apply_threshold(self, threshold):
        # Zero out pixels below the threshold
        self.thresholded_heatmap[self.heatmap <= threshold] = 0

        # Return thresholded map
        return self.heatmap

    def compute_heatmap(self, bbox_list, threshold=1):
        # print('Compute Heatmap Treshold: ', threshold)

        self.heatmap = np.zeros(self.img_shape).astype(np.float)
        self.add_heat(bbox_list)
        self.apply_threshold(threshold)
        return self.thresholded_heatmap

    def compute_heatmapN(self, bbox_list, threshold=10):
        self.multi_boxes_list.append(bbox_list)
        # print("Threshold: ", threshold)
        # print("Multi Boxes List:", self.multi_boxes_list)

        self.heatmap.fill(0)

        for bbox_list in self.multi_boxes_list:
            for box in bbox_list:
                self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # print("Length of DQ", len(self.multi_boxes_list))
        self.thresholded_heatmap.fill(0)
        self.thresholded_heatmap[self.heatmap > threshold] = 1
        return self.thresholded_heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()

            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

        # Return the image
        return img
