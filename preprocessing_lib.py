import numpy as np
import os
import json
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from resnet import FeaturesExtractor


def balance_bigger_labels(img_list, categ_array):
    label_array = np.zeros(14)
    img_list_unique = []
    categ_unique = []

    #Randomizzo l'ordine dei campioni prima del filtraggio
    index = np.random.choice(img_list.shape[0], len(img_list), replace=False)
    img_list = img_list[index]
    categ_array = categ_array[index]

    for i in range(len(img_list)):
        label = categ_array[i]
        img = img_list[i]
        #Soglia massima di campioni per ogni label
        if label_array[label] < 8000:
            img_list_unique.append(img)
            categ_unique.append(label)
            label_array[label] += 1

    print(len(img_list_unique))

    print(label_array)
    return img_list_unique, categ_unique


def filter_by_occlusion(img_list, df):
    img_list_unique = []
    categ_unique = []
    label_array = np.zeros(14)

    for i in range(len(img_list)):
        occlusion = df.iloc[i]['occlusion']
        category = df.iloc[i]['category']

        #Tengo solamente i campioni con l'occlusione migliore
        #Inoltre mantengo anche tutte le immagini delle categorie meno rappresentate
        if occlusion == 1 or category == 3 or category == 6 or category == 13:
            img_list_unique.append(img_list[i])
            categ_unique.append(category)
            label_array[category] += 1

    '''
    np.save("img_list_occlusion_bal.npy", img_list_unique)
    np.save("categ_occlusion_bal.npy", categ_unique)
    '''
    print(len(img_list_unique))
    print(label_array)

    return img_list_unique, categ_unique


def hog_feat_extractor():
    anno_folder = "train/annos/"
    img_folder = "train/image/"
    img_array = []
    categ_array = []

    j = 1
    for filename in os.listdir(anno_folder):

        with open(anno_folder + filename) as json_file:
            data = json.load(json_file)
            filtered_data = {k: v for k, v in data.items() if k.startswith('item')}
            filename = filename.split('.')[0] + ".jpg"

            for key, v in filtered_data.items():
                bbox = v['bounding_box']
                x, y, w, h = [bbox[i] for i in range(0, 4)]
                category = v['category_id']
                categ_array.append(int(category))
                img = imread(img_folder + filename)
                img = img[int(y):int(h), int(x):int(w), :]
                #Nel caso di bbox = [0, 0, 0, 0]
                if img.size < 10:
                    print("ERROR: " + str(j))
                    continue
                img = resize(img, (128, 64))
                fd, _ = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
                img_array.append(fd)
        j += 1

    imgnp = np.array(img_array)
    img_list = np.stack(imgnp)

    '''
    np.save("img_list_hog.npy", img_list)
    np.save("categ_array_hog.npy", categ_array)
    '''

    return img_list, categ_array


def neural_feat_extractor():
    #ESTRAZIONE FEATURE NEURALI
    anno_folder = "train/annos/"
    img_folder = "train/image/"
    img_array = []
    categ_array = []

    # Extractor initialization
    extractor = FeaturesExtractor()
    i = 1
    img_array = []
    for filename in os.listdir(anno_folder):
        with open(anno_folder + filename) as json_file:
            data = json.load(json_file)
            v = data['item1']
            bbox = v['bounding_box']
            x, y, w, h = [bbox[i] for i in range(0, 4)]
            category = v['category_id']
            categ_array.append(int(category))

        filename = filename.split('.')[0] + ".jpg"
        img = imread(img_folder + filename)
        img = img[int(y):int(h), int(x):int(w), :]
        img = resize(img, (64, 64))
        fd = extractor.getFeatures(img)
        img_array.append(fd)
        i += 1

    img_list = np.stack(img_array)
    #np.save("img_list_neural.npy", img_list)

    return img_list, categ_array


def data_preparation(img_list, categ_array):

    x_data = []
    y_data = []

    for i in range(len(img_list)):
        img = img_list[i]
        x_data.append(img)
        y_data.append(categ_array[i])

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    #Suddivisione dati tra Train e Test set
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)


    np.save("x_train.npy", x_train)
    np.save("x_test.npy", x_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    return x_train, x_test, y_train, y_test


def df_from_dataset(anno_folder):
    df_tot = pd.DataFrame(columns=['category', 'occlusion'])
    i = 1
    for filename in os.listdir(anno_folder):
        with open(anno_folder + filename) as json_file:
            data = json.load(json_file)
            filtered_data = {k: v for k, v in data.items() if k.startswith('item')}

            for key, v in filtered_data.items():
                df = pd.DataFrame([[v['category_id'], v['occlusion']]], columns=['category', 'occlusion'])
                df_tot = df_tot.append(df)
        if i % 100:
            print(i)
        i += 1

    df_tot = df_tot.reset_index(drop=True)

    df_tot.to_pickle("df_train.pkl")
