import numpy as np
from collections import Counter
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
    Normalizing data with StandardScaler 
'''
def data_scaler(x_train, x_test):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test

'''
    KNN Classifier
'''
def knn_classifier(test_images, x_train, x_test, y_train, y_test, k, metric):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    model = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Calculating Accuracy
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print('Accuracy: ' + "{0:.2f}".format(accuracy * 100) + '%')

    # Showing confusion matrix (textual) and classification report
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Plotting qualitative results
    random.seed()
    samples_to_show = [random.randint(0, y_test.shape[0]-1) for i in range(4)]

    fig = plt.figure(figsize=(8, 8))
    for i in range(len(samples_to_show)):
        sub = fig.add_subplot(1, len(samples_to_show), i+1)
        filename = test_images[samples_to_show[i]]
        img = mpimg.imread("train/image/" + filename + ".jpg")

        # Printing prediction and ground truth
        txt = "Predicted: " + str(y_pred[samples_to_show[i]]) + "\n" + "Ground Truth: " + str(y_test[samples_to_show[i]]) + "\n"
        sub.text(.5, .05, txt, ha='left')
        plt.axis("off")
        plt.imshow(img)

        # Saving on file
        plt.savefig("knn_k" + str(k) + "_" + str(metric))
    plt.clf()
