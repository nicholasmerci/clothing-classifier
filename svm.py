import numpy as np
from joblib import dump, load
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
    SVM Parameters tuning
'''
def parameters_tuning(x_train, x_test, y_train, y_test, tuned_parameters, scores):
    # Creating a subset of the training set and scaling its features
    train_subset = 3000
    test_subset = 300

    x_train = x_train[:train_subset]
    x_test = x_train[:test_subset]
    y_train = y_train[:train_subset]
    y_test = y_train[:test_subset]

    # Scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    for score in scores:
        print("Tuning hyper-parameters for %s\n" % score)

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:\n")
        print(clf.best_params_)
        print("Grid scores on development set:\n")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:\n")
        print("The model is trained on the full development set.\n")
        print("The scores are computed on the full evaluation set.\n")

        y_true, y_pred = y_test, clf.predict(x_test)
        # Showing best parameters
        print(classification_report(y_true, y_pred))


def svm_linear(test_images, x_train, x_test, y_train, y_test, C, max_iteration):

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

    results_calculator(test_images, models, x_train, x_test, y_train, y_test, kernel, max_iteration)


def svm_poly(test_images, x_train, x_test, y_train, y_test, C, degree, max_iteration):

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

    results_calculator(test_images, models, x_train, x_test, y_train, y_test, kernel, max_iteration)


def svm_rbf(test_images, x_train, x_test, y_train, y_test, C, gamma, max_iteration):

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

    results_calculator(test_images, models, x_train, x_test, y_train, y_test, kernel, max_iteration)

'''
    Calculating SVM classification results
'''
def results_calculator(test_images, models, x_train, x_test, y_train, y_test, kernel, iterations):

    for i in range(13):
        models[i].fit(x_train, y_train == i + 1)
        print(i)

    predicted_scores = []
    for i in range(13):
        predicted_scores.append(models[i].predict_proba(x_test)[:, 1])

    predicted_scores = np.asarray(predicted_scores)
    predicted = np.argmax(predicted_scores, axis=0)

    # Building confusion matrix
    cmc = np.zeros((13, 13))

    for pr, y_te in zip(predicted, y_test):
        cmc[y_te-1, pr] += 1.0

    # Calculating accuracy
    accuracy = np.sum(cmc.diagonal())/np.sum(cmc)

    # Calculating precision and recall
    precision = []
    recall = []
    for i in range(13):
        if cmc[i, i] != 0:
            precision.append(cmc[i, i] / np.sum(cmc[:, i]))
            recall.append(cmc[i, i] / np.sum(cmc[i, :]))
    precision = np.mean(np.asarray(precision))
    recall = np.mean(np.asarray(recall))

    print('Accuracy: ' + "{0:.2f}".format(accuracy*100) + '%')
    print(confusion_matrix(y_test, predicted))
    print('Avg precision: ' + "{0:.2f}".format(precision))
    print('Avg recall: ' + "{0:.2f}".format(recall))

    # Plotting qualitative results
    random.seed()
    samples_to_show = [random.randint(0, y_test.shape[0]-1) for i in range(4)]

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    fig = plt.figure(figsize=(8, 8))
    for i in range(len(samples_to_show)):
        sub = fig.add_subplot(1, len(samples_to_show), i+1)
        filename = test_images[samples_to_show[i]]
        img = mpimg.imread("train/image/" + filename + ".jpg")

        # Printing prediction and ground truth
        txt = "Predicted: " + str(predicted[samples_to_show[i]]+1) + "\n" + "Ground Truth: " + str(y_test[samples_to_show[i]]) + "\n"
        sub.text(.5, .05, txt, ha='left')
        plt.imshow(img)

        # Saving on file
        plt.savefig("svm_kernel_"+str(kernel)+"_iter_"+str(iterations))
    plt.clf()
