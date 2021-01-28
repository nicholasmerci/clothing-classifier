import pandas as pd
from preprocessing_lib import *

#Dataframe contenente tutte le info sul dataset
df = pd.read_pickle("df300.pkl")
''' Estrazione feature '''
#img_list, categ_array = hog_feat_extractor()

img_list = np.load("img_list_TOT.npy")
categ_array = np.load("categ_array_TOT.npy")

print(img_list.shape, categ_array.shape)
#img_list, categ_array = neural_feat_extractor()
print("hog")

#Filtro in base al livello di occlusione
img_list, categ_array = filter_by_occlusion(img_list, df)
print("occlusion")
img_list = np.asarray(img_list)
categ_array = np.asarray(categ_array)

#Cerco di bilanciare le classi maggiormente rappresentate
img_list, categ_array = balance_bigger_labels(img_list, categ_array)
print("bigger")

#Preparo i dati ad essere utilizzati con KNN e SVM
data_preparation(img_list, categ_array)
