from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
import scipy as sp


def Kmeans_clustering(
    dataframe,
    n_clusters,
    init="k-means++",
    n_init=10,
    max_iter=300,
    tol=0.0001,
    random_state=200,
):
    kmEngine = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    kmEngine.fit(dataframe)
    clusters_km = kmEngine.predict(dataframe)
    dataframe["cluster_number"] = clusters_km
    return dataframe, clusters_km


def print_content_cluster(dataframe, num):
    return dataframe[dataframe["cluster_number"] == num].index.tolist()


def print_all_content_cluster(dataframe):
    for num in range(dataframe["cluster_number"].max() + 1):
        print("Cluster number", num)
        print(print_content_cluster(dataframe, num))
        print("______________________________________________________")


def hierarchical_clustering(dataframe, nb_clust, method="complete", metric="euclidean"):
    Z = sch.linkage(dataframe, method=method, metric="euclidean")
    clusters = sch.fcluster(Z, nb_clust, criterion="maxclust")
    return Z, clusters


def dbscan_clustering(df, eps, min_samples):
    db_clust = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
    clusters_db = db_clust.labels_
    return clusters_db


def plot_knn_dist(df, n_neighbors):
    # n_neighbors = 5 as kneighbors function returns distance of point to itself (i.e. first column will be zeros)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(df)
    # Find the k-neighbors of a point
    neigh_dist, neigh_ind = nbrs.kneighbors(df)
    # sort the neighbor distances (lengths to points) in ascending order
    # axis = 0 represents sort along first axis i.e. sort along row
    sort_neigh_dist = np.sort(neigh_dist, axis=0)
    k_dist = sort_neigh_dist[:, 5]
    plt.plot(
        k_dist,
        "bx-",
        color="red",
        linewidth=1,
        markersize=3,
        label="k-NN distance",
        alpha=0.5,
    )
    plt.ylabel("k-NN distance")
    plt.xlabel("Sorted observations (th NN)")
    plt.show()


def drawSSEPlotForKMeans(
    df, nmax_clusters, max_iter=300, tol=1e-04, init="k-means++", n_init=10
):
    inertia_values = []
    for i in range(2, nmax_clusters + 1):
        km = KMeans(
            n_clusters=i,
            max_iter=max_iter,
            tol=tol,
            init=init,
            n_init=n_init,
            random_state=200,
        )
        km.fit_predict(df)
        inertia_values.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(
        range(2, nmax_clusters + 1),
        inertia_values,
        "bx-",
        color="red",
        linewidth=1,
        markersize=3,
        label="k-NN distance",
        alpha=0.5,
    )
    plt.xlabel("No. of Clusters")
    plt.ylabel("SSE / Inertia")
    plt.show()


def silhouette_score_plot(df, nmax_clusters, r):
    sil_scores = []
    for i in range(2, nmax_clusters + 1):
        sil = 0
        for k in range(r):
            clusters = KMeans(n_clusters=i, init="random", n_init=1).fit(df)
            sil += silhouette_score(
                df, clusters.labels_, metric="euclidean", sample_size=None
            )
        sil_scores.append(sil / r)
    plt.plot(
        range(2, nmax_clusters + 1),
        sil_scores,
        "bx-",
        color="red",
        linewidth=1,
        markersize=3,
        label="k-NN distance",
        alpha=0.5,
    )
    plt.xlabel("No. of Clusters")
    plt.ylabel("Silhouette Score")
    plt.show()


def compute_sse_and_sil(df, n_iter):
    sil_scores, sse_list = [], []
    for i in range(n_iter):
        km = KMeans(n_clusters=4, init="random", n_init=1)
        km.fit(df)
        labels = km.predict(df)
        sse_list.append(np.sqrt(km.inertia_))
        sil_scores.append(silhouette_score(df, labels, metric="euclidean"))
    return sse_list, sil_scores


def compare_entropy(clusters1, clusters2):
    crosstab = pd.crosstab(clusters1, clusters2)
    proba = crosstab.values / crosstab.values.sum(
        axis=1, keepdims=True
    )  # divide each element of a row by the sum of the row
    entropy = [sp.stats.entropy(row, base=2) for row in proba]
    return entropy, crosstab
