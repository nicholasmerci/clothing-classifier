import numpy as np
from collections import Counter
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def data_scaler(x_train, x_test):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test


def knn_classifier(test_images, x_train, x_test, y_train, y_test, k, metric):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print(Counter(y_train))
    print(Counter(y_test))

    model = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(x_train, y_train)
    y_pred = model.predict(x_test)

    #Calcolo l'accuratezza
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print('Accuratezza del classificatore: ' + "{0:.2f}".format(accuracy * 100) + '%')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


    random.seed()
    images_to_print = [random.randint(0, 11038) for i in range(4)]

    print('predicted:')
    for i in images_to_print:
        print(y_pred[i])

    print('ground truth:')

    for i in images_to_print:
        print(y_test[i])

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    fig = plt.figure(figsize=(8, 8))
    for i in range(len(images_to_print)):
        sub = fig.add_subplot(1, len(images_to_print), i+1)
        filename = test_images[images_to_print[i]]
        img = mpimg.imread("train/image/" + filename + ".jpg")
        txt = "Predicted: " + str(y_pred[images_to_print[i]]) + "\n" + "Ground Truth: " + str(y_test[images_to_print[i]]) + "\n"
        sub.text(.5, .05, txt, ha='left')
        plt.axis("off")
        plt.imshow(img)
        plt.savefig("knn"+str(timestr))
    plt.clf()
