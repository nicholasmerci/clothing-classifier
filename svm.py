import numpy as np
from joblib import dump, load
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


def parameters_tuning(x_train, x_test, y_train, y_test, tuned_parameters, scores):

    #Prendo un subset del dataset di training e ne scalo le feature
    x_train = x_train[:3000]
    x_test = x_train[:300]
    y_train = y_train[:3000]
    y_test = y_train[:300]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(x_test)
        print(classification_report(y_true, y_pred))


def svm_linear(x_train, x_test, y_train, y_test, C, max_iteration):

    kernel = 'linear'

    models = [SVC(C=C, kernel=kernel, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, max_iter=max_iteration, probability=True)]

    # dump(models, 'modelSVC.joblib')
    # models = load('modelSVC.joblib')

    results_calculator(models, x_train, x_test, y_train, y_test)


def svm_poly(x_train, x_test, y_train, y_test, C, degree, max_iteration):

    kernel = 'poly'

    models = [SVC(C=C, kernel=kernel, degree=degree, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, degree=degree, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, degree=degree, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, degree=degree, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, degree=degree, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, degree=degree, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, degree=degree, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, degree=degree, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, degree=degree, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, degree=degree, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, degree=degree, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, degree=degree, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, degree=degree, max_iter=max_iteration, probability=True)]

    # dump(models, 'modelSVC.joblib')
    # models = load('modelSVC.joblib')

    results_calculator(models, x_train, x_test, y_train, y_test)


def svm_rbf(x_train, x_test, y_train, y_test, C, gamma, max_iteration):

    kernel = 'rbf'

    models = [SVC(C=C, kernel=kernel, gamma=gamma, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, gamma=gamma, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, gamma=gamma, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, gamma=gamma, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, gamma=gamma, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, gamma=gamma, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, gamma=gamma, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, gamma=gamma, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, gamma=gamma, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, gamma=gamma, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, gamma=gamma, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, gamma=gamma, max_iter=max_iteration, probability=True),
              SVC(C=C, kernel=kernel, gamma=gamma, max_iter=max_iteration, probability=True)]

    # dump(models, 'modelSVC.joblib')
    # models = load('modelSVC.joblib')

    results_calculator(models, x_train, x_test, y_train, y_test)


def results_calculator(models, x_train, x_test, y_train, y_test):

    for i in range(13):
        models[i].fit(x_train, y_train == i + 1)
        print(i)

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
    print(confusion_matrix(y_test, predicted))
    print('Precisione media del classificatore: ' + "{0:.2f}".format(precision))
    print('Recall media del classificatore: ' + "{0:.2f}".format(recall))