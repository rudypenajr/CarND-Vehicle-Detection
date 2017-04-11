from CarClassifier import CarClassifier
from HeatMapper import HeatMapper
from scipy.ndimage.measurements import label
from lesson_functions import get_hog_features, bin_spatial, color_hist, draw_labeled_bboxes, convert_color
import cv2
import numpy as np

class VehicleTracker():
    def __init__(self, image_shape, heatmap_threshold, retrain=False):
        self.car_classifier = CarClassifier(train=retrain)
        self.heatmap = HeatMapper(image_shape)
        self.heatmap_threshold = heatmap_threshold
        self.scales = [1]

        self.boxed_img = None
        self.box_list = None

    def pipeline(self, img, threshold=1, debug=False):
        y_start = 400
        y_stop = 656
        scale = 1.5
        svc_threshold = 0.0
        orient = 9
        pix_per_cell = 8
        cell_per_block = 2
        spatial_size = (32, 32)
        hist_bins = 32
        svc = self.car_classifier.clf
        X_scaler = self.car_classifier.feature_scaler

        # Draw Boxes
        boxed_img, box_list = self.detect_cars(img, y_start, y_stop, scale, svc, X_scaler,
                                             orient, pix_per_cell, cell_per_block,
                                             spatial_size, hist_bins, svc_threshold)

        # Apply Heatmap
        heatmap = self.heatmap.compute_heatmap(box_list,threshold)
        labels = label(heatmap)
        final_img = self.heatmap.draw_labeled_bboxes(np.copy(img), labels)

        return boxed_img, box_list, heatmap, final_img

    def video_pipeline(self, img, threshold=1, debug=False):
        y_start = 400
        y_stop1 = 656
        scale1 = 1.5
        svc_threshold1 = 0.0

        y_stop2 = 500
        scale2 = 0.75
        svc_threshold2 = 0

        orient = 9
        pix_per_cell = 8
        cell_per_block = 2
        spatial_size = (32, 32)
        hist_bins = 32
        threshold = self.heatmap_threshold
        svc = self.car_classifier.clf
        X_scaler = self.car_classifier.feature_scaler

        # Draw Boxes
        init_img, box_list1 = self.detect_cars(img,
                                    y_start, y_stop1, scale1, svc, X_scaler,
                                    orient, pix_per_cell, cell_per_block,
                                    spatial_size, hist_bins,svc_threshold1)
        init_img, box_list2 = self.detect_cars(img,
                                    y_start, y_stop2, scale2, svc, X_scaler,
                                    orient, pix_per_cell, cell_per_block,
                                    spatial_size, hist_bins,svc_threshold2)
        box_list = box_list1 + box_list2

        # Apply Heatmap
        heatmap = self.heatmap.compute_heatmap(box_list,threshold)
        labels = label(heatmap)
        final_img = self.heatmap.draw_labeled_bboxes(np.copy(img), labels)

        return final_img

    def detect_cars(self, img, y_start, y_stop, scale, svc, X_scaler, orient,
                    pix_per_cell, cell_per_block, spatial_size, hist_bins, svc_decision_threshold):

        draw_img = np.copy(img)

        img = img.astype(np.float32) / 255.0
        area_to_search = img[y_start:y_stop,:,:]
        ctrans_tosearch = convert_color(area_to_search, conv="RGB2YCrCb")
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1
        nfeat_per_block = orient * cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        car_boxes = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1 and svc.decision_function(test_features) > svc_decision_threshold: #if car found
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)

                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+y_start),(xbox_left+win_draw,ytop_draw+win_draw+y_start),(0,0,255),6)
                    startx = xbox_left
                    starty = ytop_draw+y_start
                    endx = xbox_left+win_draw
                    endy = ytop_draw+win_draw+y_start
                    car_boxes.append(((startx,starty),(endx,endy)))
        return draw_img, car_boxes
