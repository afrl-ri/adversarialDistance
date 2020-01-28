##
##   Code from Bansal and Weld paper aiweb.cs.washington.edu/ai/unkunk18
##
from __future__ import division
import os

from sklearn.cluster import KMeans
import numpy as np

#import config


def cluster(algo, inputs, k, conf=None, num_processes=-1):
    clusters = None
    # if os.path.isfile(config.clusterfile):
    # print "loading", config.clusterfile
    # return np.load(config.clusterfile)
    if algo == "kmeans":
        kmeans = KMeans(
            n_clusters=k, random_state=1, n_jobs=num_processes).fit(inputs)
        clusters = kmeans.labels_
    elif algo == "kmeans_conf":
        assert conf is not None
        # cluster by confidence
        conf = conf.reshape((inputs.shape[0], 1))
        conf_k = elbow(conf, 10, plot=False)
 #       print "conf_k", conf_k
        conf_kmeans = KMeans(
            n_clusters=conf_k, random_state=1, n_jobs=num_processes).fit(conf)
        clusters = conf_kmeans.labels_
    elif algo == "kmeans_both":
        assert conf is not None
        clusters = np.array([None for i in range(inputs.shape[0])])

        # cluster by confidence
        conf = conf.reshape((inputs.shape[0], 1))
        conf_k = elbow(conf, 5, plot=False)
 #       print "conf_k", conf_k
        conf_kmeans = KMeans(n_clusters=conf_k, random_state=1, n_jobs=num_processes).fit(conf)
        conf_clusters = conf_kmeans.labels_
 #       print conf_kmeans.cluster_centers_

        offset = 0
        for i in range(conf_k):
            idxs = np.nonzero(conf_clusters == i)[0]
            # cluster by features
            i_k = elbow(inputs[idxs, :], 5, plot=False)
  #          print "%d_k" % i, idxs.shape[0], i_k, offset
            i_kmeans = KMeans(
                n_clusters=i_k, random_state=1,
                n_jobs=num_processes).fit(inputs[idxs, :])
            clusters[idxs] = i_kmeans.labels_ + offset
            offset += i_k
 #       print np.unique(clusters)
    elif algo == "kmeans_both3":
        assert conf is not None
        clusters = np.array([None for i in range(inputs.shape[0])])

        # cluster by confidence
        conf = conf.reshape((inputs.shape[0], 1))
        bins = np.array([0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
        conf_k = bins.shape[0] - 1
        conf_clusters = np.digitize(conf, bins, right=True) - 1
  #      print conf_clusters

        offset = 0
        for i in range(conf_k):
            idxs = np.nonzero(conf_clusters == i)[0]
  #          print idxs.shape
            # cluster by features
            i_k = elbow(inputs[idxs, :], 5, plot=False)
  #          print "%d_k" % i, idxs.shape[0], i_k, offset
            i_kmeans = KMeans(
                n_clusters=i_k, random_state=1,
                n_jobs=num_processes).fit(inputs[idxs, :])
            clusters[idxs] = i_kmeans.labels_ + offset
            offset += i_k
  #      print np.unique(clusters)
    elif algo == "kmeans_both2":
        assert conf is not None
        clusters = np.array([None for i in range(inputs.shape[0])])

        # cluster by confidence
        conf = conf.reshape((inputs.shape[0], 1))
        conf_k = elbow(conf, 10, plot=False)
  #      print "conf_k", conf_k
        conf_kmeans = KMeans(
            n_clusters=conf_k, random_state=1, n_jobs=num_processes).fit(conf)
        conf_clusters = conf_kmeans.labels_
   #     print conf_kmeans.cluster_centers_

        feature_k = elbow(inputs, 10, plot=False)
    #    print "feature_k", feature_k
        feature_kmeans = KMeans(
            n_clusters=feature_k, random_state=1,
            n_jobs=num_processes).fit(inputs)
        feature_clusters = feature_kmeans.labels_
        offset = 0
        for i in range(conf_k):
            idxs = np.nonzero(conf_clusters == i)[0]
            f_clust = feature_clusters[idxs]
            uniq = np.unique(f_clust)
            for j, c in enumerate(uniq):
                common_idxs = np.nonzero(
                    np.multiply(conf_clusters == i, feature_clusters == c))[0]
                clusters[common_idxs] = offset + j
            offset += uniq.shape[0]

      #  print np.unique(clusters)

    # print "saving", config.clusterfile
    # np.save(config.clusterfile, clusters)
    return clusters


def elbow(inputs, n, outfile=None, num_processes=-1, plot=True):
    Ks = range(2, n + 1)

    def sse(labels, centers):
        center_array = centers[labels]
        return np.sum(np.power(inputs - center_array, 2)) / inputs.shape[0]

    km = [KMeans(n_clusters=i, n_jobs=num_processes) for i in Ks]
    for i in range(len(km)):
        km[i].fit(inputs)
    score = [
        sse(km[i].labels_, km[i].cluster_centers_) for i in range(len(km))
    ]
    score_diff = np.diff(score)
    diff_ratio = score_diff[1:] / score_diff[:-1]
    best_k = Ks[np.argmax(diff_ratio)] + 1
    if plot is True:
        plt.grid()
        plt.plot(Ks, score, marker="o")
        plt.savefig(outfile)
        plt.close()
    return best_k


def cluster_unkunk_dist(k, clusters, utility_model):
#    print k
    arms = [[] for i in range(k)]
    arm_unkunks = [[] for i in range(k)]
 #   print clusters.shape
 #   print utility_model.num_inputs
    for i in range(utility_model.num_inputs):
        arms[clusters[i]].append(i)
        if utility_model._is_unkunk(i):
            arm_unkunks[clusters[i]].append(i)
    sizes = np.array([len(a) for a in arms])
    unkunk_sizes = np.array([len(a) for a in arm_unkunks])
    num_unkunks = utility_model._get_num_unkunks()
 #   print "cluster sizes", sizes
 #   print "cluster unkunk size", unkunk_sizes
 #   print "unkunk ratio", unkunk_sizes / sizes
    entropy = -1 * np.sum(
        np.log((unkunk_sizes + 1) / num_unkunks) * unkunk_sizes / num_unkunks)
  #  print "entropy", k, entropy
