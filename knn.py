import numpy as np
from collections import Counter

def data_scaler(x_train, x_test):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test


def knn_classifier(x_train, x_test, y_train, y_test, k, metric):
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
