import numpy as np
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


x_train = np.load('x_train_occ_PCA80.npy')
x_test = np.load('x_test_occ_PCA80.npy')

y_train = np.load('y_train_occ.npy')
y_test = np.load('y_test_occ.npy')

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
x_train = x_train[:5000]
x_test = x_train[:500]
y_train = y_train[:5000]
y_test = y_train[:500]


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


kernel = 'linear'
max_iteration = 10

models = [SVC(kernel=kernel, max_iter=max_iteration, probability=True),
          SVC(kernel=kernel, max_iter=max_iteration, probability=True),
          SVC(kernel=kernel, max_iter=max_iteration, probability=True),
          SVC(kernel=kernel, max_iter=max_iteration, probability=True),
          SVC(kernel=kernel, max_iter=max_iteration, probability=True),
          SVC(kernel=kernel, max_iter=max_iteration, probability=True),
          SVC(kernel=kernel, max_iter=max_iteration, probability=True),
          SVC(kernel=kernel, max_iter=max_iteration, probability=True),
          SVC(kernel=kernel, max_iter=max_iteration, probability=True),
          SVC(kernel=kernel, max_iter=max_iteration, probability=True),
          SVC(kernel=kernel, max_iter=max_iteration, probability=True),
          SVC(kernel=kernel, max_iter=max_iteration, probability=True),
          SVC(kernel=kernel, max_iter=max_iteration, probability=True)]

for i in range(13):
    models[i].fit(x_train, y_train == i+1)

dump(models, 'modelSVC.joblib')
#models = load('modelSVC.joblib')

predicted_scores = []
for i in range(13):
    predicted_scores.append(models[i].predict_proba(x_test)[:, 1])

predicted_scores = np.asarray(predicted_scores)
predicted = np.argmax(predicted_scores, axis=0)

cmc = np.zeros((13, 13))

for pr, y_te in zip(predicted, y_test):
    cmc[y_te-1, pr] += 1.0

accuracy = np.sum(cmc.diagonal())/np.sum(cmc)

precision = []
recall = []
for i in range(13):
    if cmc[i, i] != 0:
        precision.append(cmc[i, i] / np.sum(cmc[:, i]))
        recall.append(cmc[i, i] / np.sum(cmc[i, :]))
precision = np.mean(np.asarray(precision))
recall = np.mean(np.asarray(recall))

print('Accuratezza del classificatore: ' + "{0:.2f}".format(accuracy*100) + '%')
print('Precisione media del classificatore: ' + "{0:.2f}".format(precision))
print('Recall media del classificatore: ' + "{0:.2f}".format(recall))
