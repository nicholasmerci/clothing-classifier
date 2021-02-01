import pandas as pd
from preprocessing_lib import *
from knn import *
from pca import *
from svm import *


#Dataframe contenente tutte le info sul datasetù

anno_folder = "train/annos/"
df = df_from_dataset(anno_folder)
#df = pd.read_pickle("df_train.pkl")

#Estrazione HOG feature
img_list, categ_array = hog_feat_extractor()

'''img_list = np.load("img_list.npy")
categ_array = np.load("categ_array.npy")
'''

#Estrazione HOG feature
#img_list, categ_array = neural_feat_extractor()

#Filtro in base al livello di occlusione
img_list, categ_array = filter_by_occlusion(img_list, df)
img_list = np.asarray(img_list)
categ_array = np.asarray(categ_array)

#Cerco di bilanciare le classi maggiormente rappresentate
img_list, categ_array = balance_bigger_labels(img_list, categ_array)

#Preparo i dati ad essere utilizzati con KNN e SVM
x_train, x_test, y_train, y_test = data_preparation(img_list, categ_array)

#Normalizzo le feature
x_train, x_test = data_scaler(x_train, x_test)

#Riduco dimensionalità feature tramite PCA
n_components = .75
x_train, x_test = pca(x_train, x_test, n_components)

#Creo un classificatore con KNN
k = 1
metric = 'cosine'
knn_classifier(x_train, x_test, y_train, y_test, k, metric)

#Tuning dei parametri per SVM
#Calcolo i migliori parametri per precision e recall
scores = ['precision', 'recall']

#tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#tuned_parameters = [{'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'degree': [2, 3, 4, 5, 6]}]
tuned_parameters = [{'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}]

parameters_tuning(x_train, x_test, y_train, y_test, tuned_parameters, scores)

#Creo un classificatore con SVM
#Setto i parametri necessari
C = 10
degree = 2
gamma = 0.001
max_iteration = 10

#svm_linear(x_train, x_test, y_train, y_test, C, max_iteration)
#svm_poly(x_train, x_test, y_train, y_test, C, degree, max_iteration)
svm_rbf(x_train, x_test, y_train, y_test, C, gamma, max_iteration)