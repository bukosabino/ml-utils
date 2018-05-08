# -*- coding: utf-8 -*-

"""Some utils functions to do Feature Engineering:
Simple and computationally efficient way to reduce the dimensionality of the
data.
"""

from sklearn.decomposition import *
from sklearn.random_projection import *


# Wrapper
def add_all_features(df, features):
    n_comp = 3
    df = add_mspca_features(df, n_comp, features)
    df = add_spca_features(df, n_comp, features)
    df = add_ipca_features(df, n_comp, features)
    df = add_pca_features(df, n_comp, features)
    df = add_tsvd_features(df, n_comp, features)
    df = add_ica_features(df, n_comp, features)
    df = add_grp_features(df, n_comp, features)
    df = add_srp_features(df, n_comp, features)
    df = add_fa_features(df, n_comp, features)
    return df


# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchSparsePCA.html
def add_mspca_features(df, n_comp, features):
    mspca = MiniBatchSparsePCA(n_components=n_comp, random_state=17)
    mspca_results = mspca.fit_transform(df[features])
    for i in range(1, n_comp + 1):
        df['feature_mspca_' + str(i)] = mspca_results[:, i - 1]
    return df


# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html
def add_spca_features(df, n_comp, features):
    spca = SparsePCA(n_components=n_comp, random_state=17)
    spca_results = spca.fit_transform(df[features])
    for i in range(1, n_comp + 1):
        df['feature_spca_' + str(i)] = spca_results[:, i - 1]
    return df


# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
def add_ipca_features(df, n_comp, features):
    ipca = IncrementalPCA(n_components=n_comp)
    ipca_results = ipca.fit_transform(df[features])
    for i in range(1, n_comp + 1):
        df['feature_ipca_' + str(i)] = ipca_results[:, i - 1]
    return df


# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
def add_pca_features(df, n_comp, features):
    pca = PCA(n_components=n_comp, random_state=17)
    pca_results = pca.fit_transform(df[features])
    for i in range(1, n_comp + 1):
        df['feature_pca_' + str(i)] = pca_results[:, i - 1]
    return df


# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
def add_tsvd_features(df, n_comp, features):
    tsvd = TruncatedSVD(n_components=n_comp, random_state=17)
    tsvd_results = tsvd.fit_transform(df[features])
    for i in range(1, n_comp + 1):
        df['feature_tsvd_' + str(i)] = tsvd_results[:, i - 1]
    return df


# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
def add_ica_features(df, n_comp, features):
    ica = FastICA(n_components=n_comp, random_state=17)
    ica_results = ica.fit_transform(df[features])
    for i in range(1, n_comp + 1):
        df['feature_ica_' + str(i)] = ica_results[:, i - 1]
    return df


# http://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html
def add_grp_features(df, n_comp, features):
    grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=17)
    grp_results = grp.fit_transform(df[features])
    for i in range(1, n_comp + 1):
        df['feature_grp_' + str(i)] = grp_results[:, i - 1]
    return df


# http://scikit-learn.org/stable/modules/generated/sklearn.random_projection.SparseRandomProjection.html
def add_srp_features(df, n_comp, features):
    srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=17)
    srp_results = srp.fit_transform(df[features])
    for i in range(1, n_comp + 1):
        df['feature_srp_' + str(i)] = srp_results[:, i - 1]
    return df


# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html
def add_fa_features(df, n_comp, features):
    fa = FactorAnalysis(n_components=n_comp, random_state=17)
    fa_results = fa.fit_transform(df[features])
    for i in range(1, n_comp + 1):
        df['feature_fa_' + str(i)] = fa_results[:, i - 1]
    return df
