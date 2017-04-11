#references
# dumping and loading classifiers : http://scikit-learn.org/stable/modules/model_persistence.html

from lesson_functions import extract_features
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
import glob
import time


class CarClassifier:
	def __init__(self,train=False,verbose=False):
		#Load from file as training from scratch takes lot of time
		self.clf = None
		self.feature_scaler = None

		#Read number of histbins from config file
		self.color_space = 'YCrCb' #'RGB'
		self.spatial_size = (32,32)
		self.hist_bins = 32
		self.orient = 9
		self.pix_per_cell =8
		self.cell_per_block = 2
		self.hog_channel = 'ALL'
		self.spatial_feat= True
		self.hist_feat = True
		self.hog_feat = True

		#Load classifier and feature scaler from pre-saved files
		if verbose:
			print("Train:", train)
			if train == False:
				try:
					self.clf = joblib.load("./dependencies/car_classifier.pkl")
					self.feature_scaler = joblib.load("./dependencies/car_feature_scaler.pkl")
					if verbose:
						print("clf: ", self.clf)
						print("scaler:", self.feature_scaler)
				except Exception as e:
					print("exception thrown", str(e))


		#Train from scratch if classifier pkl file is not found
		if self.clf == None:
			print("Retraining Car Classifier")
			self.train()

	def normalize_features(self, vehicle_features, non_vehicle_features, vehicles, non_vehicles):
		X = np.vstack((vehicle_features, non_vehicle_features))
		X_scaler = StandardScaler().fit(X)
		scaled_X = X_scaler.transform(X)
		# X_train_final = scaled_X

		x_ones = np.ones(len(vehicles))
		y_ones = np.zeros(len(non_vehicles))
		feature_labels = np.hstack((x_ones, y_ones))
		# y_train_final = feature_labels

		return X_scaler, scaled_X, feature_labels

	def train(self):
		vehicles = glob.glob('./vehicles/**/*.png',recursive=True)
		non_vehicles = glob.glob('./non-vehicles/**/*.png',recursive=True)
		print("Length of Vehicle Dataset:", len(vehicles))
		print("Length of Non-Vehicle Dataset:", len(non_vehicles))

		# Extract Features from Vehicles/Non-Vehilces
		vehicle_features = extract_features(vehicles, color_space=self.color_space, spatial_size=self.spatial_size,
											hist_bins=self.hist_bins, orient=self.orient,
											pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
											hog_channel= self.hog_channel, spatial_feat=self.spatial_feat,
											hist_feat=self.hist_feat, hog_feat=self.hog_feat)

		non_vehicle_features = extract_features(non_vehicles, color_space=self.color_space, spatial_size=self.spatial_size,
												hist_bins=self.hist_bins, orient=self.orient,
												pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
												hog_channel= self.hog_channel, spatial_feat=self.spatial_feat,
												hist_feat=self.hist_feat, hog_feat=self.hog_feat)

		# Normalize Features
		X_scaler, X_train_final, y_train_final = self.normalize_features(vehicle_features, non_vehicle_features, vehicles, non_vehicles)

		# Split Data
		rand_state = np.random.randint(0, 100)
		X_train, X_test, y_train, y_test = train_test_split(X_train_final, y_train_final, test_size=0.2, random_state=rand_state)

		# Train Car Classifier
		t = time.time()
		clf = LinearSVC()
		clf.fit(X_train, y_train)
		t2 = time.time()
		print(round(t2-t, 2), 'Seconds to train SVC...')

		# Check the Score of the SVC
		print('Test Accuracy of SVC = ', clf.score(X_test, y_test))

		# Save the Classifier
		self.clf = clf
		self.feature_scaler = X_scaler
		print("Saving Classifier and Feature Scaler.")
		joblib.dump(clf,"./dependencies/car_classifier.pkl")
		joblib.dump(X_scaler,"./dependencies/car_feature_scaler.pkl")

		return
