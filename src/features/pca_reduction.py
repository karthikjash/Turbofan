from sklearn.decomposition import PCA

def apply_pca(x, n_components = 10):
    samples, window, sensors = x.shape
    x_flat = x.reshape(-1, sensors)
    pca = PCA(n_components=n_components)
    x_reduced = pca.fit_transform(x_flat)
    x_reduced = x_reduced.reshape(samples, window, n_components)

    return x_reduced, pca
