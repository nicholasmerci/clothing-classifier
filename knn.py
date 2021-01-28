import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

x_train = np.load('x_train2.npy')
x_test = np.load('x_test2.npy')

y_train = np.load('y_train2.npy')
y_test = np.load('y_test2.npy')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(Counter(y_train))
print(Counter(y_test))



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#==========================================================#
#PCA


from sklearn.decomposition import PCA

pca = PCA(.80)

pca.fit(x_train)

x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


np.save("x_train2PCA80.npy", x_train)
np.save("x_test2PCA80.npy", x_test)

print("Fine PCA")


#===========================================#
#KNN


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = KNeighborsClassifier(n_neighbors=1, metric='cosine').fit(x_train, y_train)
y_pred = model.predict(x_test)

#Calcolo l'accuratezza
accuracy = np.sum(y_pred == y_test) / len(y_test)
print('Accuratezza del classificatore: ' + "{0:.2f}".format(accuracy * 100) + '%')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
