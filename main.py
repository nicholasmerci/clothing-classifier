import pandas as pd
from preprocessing_lib import *
from knn import *
from pca import *
from svm import *
import os.path
from os import path

anno_folder = "train/annos/"
train_file = "df_train.pkl"

# Loading dataframe
print("Reading dataset...", end='')
if path.exists(train_file):
    print("from file...", end='')
    df = pd.read_pickle(train_file)
else:
    print("from dataset folder (it might take some minutes)...", end='')
    df = df_from_dataset(anno_folder, train_file)
print("loading completed.")

# Extracting HOG features
print("Feature Extraction...")

if (not path.exists("img_list_hog.npy")) and (not path.exists("categ_array_hog.npy")):
    print("no saved features, it might take several time.")
else:
    print("from file (it might take some minutes)...")

if (not path.exists("img_list_hog.npy")) and (not path.exists("categ_array_hog.npy")):
    img_list, categ_array = hog_feat_extractor()
else:
    img_list = np.load("img_list_hog.npy")
    categ_array = np.load("categ_array_hog.npy")
print("completed.")


# Extracting NN features
#img_list, categ_array = neural_feat_extractor()

# Filtering by occlusion 
print("Filtering dataset")
img_file_names, img_list, categ_array = filter_by_occlusion(img_list, df)
img_list = np.asarray(img_list)
img_file_names = np.asarray(img_file_names)
categ_array = np.asarray(categ_array)

# Balancing most representative classes
print("Classes Balancing")

img_file_names, img_list, categ_array = balance_bigger_labels(img_file_names, img_list, categ_array)
'''
np.save("img_file_names.npy", img_file_names)
np.save("img_list.npy", img_list)
np.save("categ_array.npy", categ_array)
'''
img_file_names = np.load("img_file_names.npy")
img_list = np.load("img_list.npy")
categ_array = np.load("categ_array.npy")

# Preparing dataset for KNN and SVM
print("Preparing data")
test_images, x_train, x_test, y_train, y_test = data_preparation(img_file_names, img_list, categ_array)

'''
    Reduction of features dimensionality
'''

# Feature Normalization
print("Feature Normalization")
x_train, x_test = data_scaler(x_train, x_test)

# Principal Component Analysis
print("PCA")
n_components = .75
x_train, x_test = pca(x_train, x_test, n_components)

'''
    KNN Classifier
'''
print("KNN")
k = 1
metric = 'cosine'
knn_classifier(test_images, x_train, x_test, y_train, y_test, k, metric)

'''
    SVM Classifier
'''
print("SVM")
# Parameters Tuning
# Finding best parameters wrt precision and recall

scores = ['precision', 'recall']

#tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#tuned_parameters = [{'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'degree': [2, 3, 4, 5, 6]}]
tuned_parameters = [{'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}]

parameters_tuning(x_train, x_test, y_train, y_test, tuned_parameters, scores)

# Setting parameters
C = 10
degree = 2
gamma = 0.001
max_iteration = 1000

#svm_linear(test_images, x_train, x_test, y_train, y_test, C, max_iteration)
#svm_poly(test_images, x_train, x_test, y_train, y_test, C, degree, max_iteration)
svm_rbf(test_images, x_train, x_test, y_train, y_test, C, gamma, max_iteration)
