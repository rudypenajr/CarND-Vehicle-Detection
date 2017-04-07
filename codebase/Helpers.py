import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Function draws boxes on an image.
    Reference: Manual Vehicle Detection.ipynb (Lesson 5)

    :param img: ndarray (Image)
    :param bboxes: Dictionary? [((x1, y1), (x2, y2)), ((,),(,)), ...]
    :param color: Tuple
    :param thick: Integer
    :return: Array (Copy of Original Image with Necessary Boxes Drawn)
    """
    # Make a copy of the image
    draw_img = np.copy(img)

    # draw each bounding box on your image copy using cv2.rectangle()
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)

    # return the image copy with boxes drawn
    return draw_img

def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    Because Template Matching is not robust for finding vehicles,
    the next best option is using raw pixel intensities as features.
    Reference: Chapter 12 - Histograms of Color.ipynb (Lesson 12)

    :param img: ndarray (Image)
    :param nbins: Integer
    :param bins_range: Integer
    :return: Vector (Color Histogram Features as a Single Feature Vector)
    """
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

    return hist_features

def bin_spatial(img, size=(32, 32)):
    """
    While it could be cumbersome to include 3 color channesl of a full resolution image,
    you can perform spatial binning on an image and still retain enough information.
    Reference: Chapter 16 - Spatial Binning of Color.ipynb (Lesson 16)

    :param img: ndarray (Image)
    :param size: Tuple
    :return: Feature Vector
    """
    # User cv2.resize().ravel() to create a feature vector
    features = cv2.resize(feature_image, size).ravel()
    return features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    Extract Histogram of Oriented Gradients for a given image.
    :param img: ndarray (Greyscale.)
    :param orient: Integer (# of orientation bins.)
    :param pix_per_cell: 2D Tuple (int, int) (Size in pixels of a cell.)
    :param cell_per_block: 2D Tuple (int, int) (Number of cells in each block.)
    :param vis: Bool (Return an image of the HOG.)
    :param feature_vec: Bool (Return the data as a feature vector by calling .ravel() on the result just before returning.)
    :return: ndarray (1D flattened array - if vis is False)
    """

    if vis == True:
        features, hog_image = hog(img, orienations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualize=vis,
                                  feature_vector=False)
        return features, hog_image
    else:
        features, hog_image = hog(img, orienations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualize=False,
                                  feature_vector=feature_vec)
        return features

def extract_features(imgs, cpace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
    """
    Function to extract features from a list of images
    :param imgs: List of Images
    :param cpace: String (Color Space)
    :param spatial_size: 2D Tuple
    :param hist_bins: Integer
    :param hist_range: 2D Tuple
    :return: List of Feature Vectors
    """

    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for file in imgs:
        # Read each img one by one
        img = mpimg.imread(file)

        # Apply Color Conversion if other than 'RGB
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(img)

        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)

        # Apply color_hist() to get color histogram feateures
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

        # Append the new feature vector to the features list
        features.append(np.concatenate(spatial_features, hist_features))

    return features