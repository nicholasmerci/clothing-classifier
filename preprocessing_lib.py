import numpy as np
import os
import json
import cv2
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from resnet import FeaturesExtractor
import random

'''
    Balancing the most representated classes
'''
def balance_bigger_labels(img_file_names, img_list, categ_array):
    # Maximum number of samples for each class
    max_sample_for_class = 8000

    label_array = np.zeros(14)
    img_list_reduced = []
    categ_reduced = []
    images_reduced = []

    # Randomizing order before filtering
    index = np.random.choice(img_list.shape[0], len(img_list), replace=False)
    img_list = img_list[index]
    categ_array = categ_array[index]
    img_file_names = img_file_names[index]

    j = 1
    for i in range(len(img_list)):
        label = categ_array[i]
        img = img_list[i]
        image = img_file_names[i]
        # Checking if max samples has been reached
        if label_array[label] < max_sample_for_class:
            img_list_reduced.append(img)
            categ_reduced.append(label)
            images_reduced.append(image)
            label_array[label] += 1
        # Printing percentage
        if j % 100 == 0:
            print(j, end='\r')
        j += 1

    return images_reduced, img_list_reduced, categ_reduced

'''
    Filtering images by occlusion
'''
def filter_by_occlusion(img_list, df):
    img_list_reduced = []
    categ_reduced = []
    img_file_names = []
    label_array = np.zeros(14)

    j = 1
    for i in range(len(img_list)):
        occlusion = df.iloc[i]['occlusion']
        category = df.iloc[i]['category']
        filename = df.iloc[i]['filename']
        # NOTE: Keeping all samples of the less represented classes
        if occlusion == 1 or category == 3 or category == 6 or category == 13:
            img_list_reduced.append(img_list[i])
            categ_reduced.append(category)
            img_file_names.append(filename)
            label_array[category] += 1
        if j % 1000 == 0:
            print(j, end='\r')
        j += 1

    '''
    np.save("img_list_occlusion_bal.npy", img_list_reduced)
    np.save("categ_occlusion_bal.npy", categ_reduced)
    '''

    return img_file_names, img_list_reduced, categ_reduced

'''
    Extracting HOG Features
'''
def hog_feat_extractor():
    anno_folder = "train/annos/"
    img_folder = "train/image/"
    img_array = []
    categ_array = []

    print("loading file list (will take a while)")
    j = 1
    for filename in os.listdir(anno_folder):
        if(j % 100 == 0):
            print(str(j), end='\r')
        with open(anno_folder + filename) as json_file:
            data = json.load(json_file)
            filtered_data = {k: v for k, v in data.items() if k.startswith('item1')} # Getting only first item of each image
            filename = filename.split('.')[0] + ".jpg"
            
            for key, v in filtered_data.items():
                bbox = v['bounding_box'] # BBOX for cropping
                x, y, w, h = [bbox[i] for i in range(0, 4)]
                category = v['category_id']
                categ_array.append(int(category))
                img = imread(img_folder + filename)

                # Cropping image
                img = img[int(y):int(h), int(x):int(w), :]

                # Skipping if bbox = [0, 0, 0, 0]
                if img.size < 10:
                    print("ERROR: " + str(j) + " bbox values are not valid.")
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

'''
    Extracting Neural Network Features
'''
def neural_feat_extractor():
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
            bbox = v['bounding_box']  # BBOX for cropping
            x, y, w, h = [bbox[i] for i in range(0, 4)]
            category = v['category_id']
            categ_array.append(int(category))

        filename = filename.split('.')[0] + ".jpg"
        img = imread(img_folder + filename)

        # Cropping image
        img = img[int(y):int(h), int(x):int(w), :]

        # Skipping if bbox = [0, 0, 0, 0]
        if img.size < 10:
            print("ERROR: " + str(j) + " bbox values are not valid.")
            continue
        img = resize(img, (64, 64))
        fd = extractor.getFeatures(img)
        img_array.append(fd)
        i += 1

    img_list = np.stack(img_array)
    #np.save("img_list_neural.npy", img_list)

    return img_list, categ_array

'''
    Splitting dataset for training
'''
def data_preparation(img_file_names, img_list, categ_array):
    x_data = []
    y_data = []
    # Setting percentage for training set
    train_percentage = 0.85

    for i in range(len(img_list)):
        img = img_list[i]
        x_data.append(img)
        y_data.append(categ_array[i])

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    test_images = np.asarray(img_file_names)

    random.seed()
    n_train_samples = len(img_list) * train_percentage
    train_index = random.sample(range(len(img_list + 1)), n_train_samples)
    tot_index = [i for i in range(len(img_list))]

    test_index = []
    for v in tot_index:
        if v not in train_index:
            test_index.append(v)

    x_train = x_data[train_index]
    y_train = y_data[train_index]
    x_test = x_data[test_index]
    y_test = y_data[test_index]
    test_images = img_file_names[test_index]

    np.save("x_train.npy", x_train)
    np.save("x_test.npy", x_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    np.save("test_images.npy", test_images)

    return test_images, x_train, x_test, y_train, y_test

'''
    Creating a dataframe of the entire dataset
'''
def df_from_dataset(anno_folder):
    df_tot = pd.DataFrame(columns=['category', 'occlusion'])
    i = 1

    # Reading from annotations folder every JSON file
    for filename in os.listdir(anno_folder):
        with open(anno_folder + filename) as json_file:
            data = json.load(json_file)
            filtered_data = {k: v for k, v in data.items() if k.startswith('item1')}

            # Getting useful info from dataset
            for key, v in filtered_data.items():
                df = pd.DataFrame([[v['category_id'], v['occlusion']]], columns=['category', 'occlusion'])
                df_tot = df_tot.append(df)
        if i % 100:
            print(i, end='\r')
        i += 1

    df_tot = df_tot.reset_index(drop=True)

    df_tot.to_pickle("df_train.pkl")

    return df_tot