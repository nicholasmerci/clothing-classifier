def pca(x_train, x_test, n_components):
    from sklearn.decomposition import PCA

    pca = PCA(n_components)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    print("Features dimensionality after pca:")
    print(x_train.shape, x_test.shape)

    return x_train, x_test
