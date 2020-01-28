##
##   Code from Bansal and Weld paper aiweb.cs.washington.edu/ai/unkunk18
##
from __future__ import division

import sklearn
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class UtilityModel(object):
    def __init__(self, inputs, labels, pred, conf, cost, gamma=0.0):
        self.num_inputs = inputs.shape[0]
        self.labels = labels
        self.pred = pred
        self.conf = conf
        self.cost = cost
        self.gamma = gamma

        self.unkunks = np.array(
            [int(self._is_unkunk(i)) for i in range(self.num_inputs)])

    def get_utility(self, samples):
        return np.sum([(self._is_unkunk(i) - self.gamma * self.cost[i])
                       for i in samples])

    def get_cand_unkunks(self):
        return [i for i in range(self.num_inputs)]

    def _is_unkunk(self, i):
        return (self.labels[i] != self.pred[i])

    def _get_unkunks(self):
        return np.nonzero(self.unkunks)[0]

    def _get_num_unkunks(self):
        return np.sum(self.unkunks)

    def get_mean_utility(self, solution, samples):
        return self.get_utility(samples) / len(samples)

    def get_uu_prob(self, samples):
        return sum(self.unkunks[samples]) / len(samples)

    def _save_unkunk_2d(self, inputs, outfile):
        pca = sklearn.decomposition.TruncatedSVD(
            n_components=2, random_state=0)
        if inputs.shape[1] > 2:
            x = pca.fit_transform(inputs)
        else:
            x = inputs
        unkunks = self._get_unkunks()
        plt.scatter(x[:, 0], x[:, 1], color="b")
        plt.scatter(x[unkunks, 0], x[unkunks, 1], color="r")
        plt.savefig(outfile)
        plt.close()

    def _save_unkunk_hist(self, outfile, delta=0.025):
        bins = []
        x = np.min(np.min(self.conf))
        while x <= 1.0:
            bins.append(x)
            x += delta
        bins = np.array(bins)

        num_bins = bins.shape[0]
        assigned_bins = np.digitize(self.conf, bins)
        unkunk_assigned_bins = np.digitize(self.conf[self._get_unkunks()],
                                           bins)

        prob = [0] * num_bins
        for i in range(num_bins):
            denom = np.sum(assigned_bins == i)
            if denom != 0:
                prob[i] = np.sum(unkunk_assigned_bins == i) / denom
        plt.plot(bins - delta / 2, prob, label="True prob", marker="x")
        plt.plot(bins, (1 - bins), label="Prior: 1 - conf", color="r")
        plt.xlabel("Conf")
        plt.ylabel("P(UU|conf)")
        plt.legend()
        plt.savefig(outfile)
        plt.close()

    def get_argmax_utility(self, samples, others, prior):
        argmax = np.argmax(prior[others] - self.gamma * self.cost[others])
        return prior[argmax], argmax


class SubmodUtilityModel(UtilityModel):
    def __init__(self, labels, pred, conf, var=0.1):
        self.labels = labels
        self.num_inputs = labels.shape[0]
        self.pred = pred
        self.conf = conf
        self.var = var

    def setup(self, inputs):
        self.unkunks = np.array(
            [int(self._is_unkunk(i)) for i in range(self.num_inputs)])
        self.distance_matrix = euclidean_distances(inputs, inputs)
       # print("pairwise done")
        self.distance_matrix = self.scale(self.distance_matrix)
       # print("std", np.std(self.distance_matrix))
        self.gauss_val = self.gaussian(self.distance_matrix, self.var)
        self.gauss_val_masked = self.gauss_val * self.unkunks.reshape(
            (1, self.num_inputs))
       # print("max possible utility", self.get_utility(self._get_unkunks()))
       # print(self.gauss_val.shape, self.gauss_val_masked.shape)

    def setup_from_file(self, filename):
        data = np.load(filename)
        self.unkunks = data["unkunks"]
        self.gauss_val = data["gauss_val"]
        self.gauss_val_masked = data["gauss_val_masked"]

    def scale(self, A):
        A = A / np.max(A)
        return A

    # def plot_dist_hist(self, outfile):
    # trui_idxs = np.triu_indices(self.distance_matrix.shape[0])
    # distance_vals = self.distance_matrix[trui_idxs]
    # plt.hist(distance_vals, bins=100, normed=True)
    # x = np.linspace(0, np.max(distance_vals), num=100)
    # plt.plot(x, self.gen_gaussian(x))
    # print "saving distance hist", outfile
    # plt.savefig(outfile)
    # plt.close()

    @classmethod
    def gaussian(cls, x, var):
        val = np.exp(-1 * np.power(x, 2.) / var)
        return val

    # def gen_gaussian(self, x):
    # trui_idxs = np.triu_indices(self.distance_matrix.shape[0])
    # distance_vals = self.distance_matrix[trui_idxs]
    # mean = np.mean(
    # sorted(distance_vals)[self.num_inputs:self.num_inputs + 100])
    # print mean
    # c = mean * 2 / 2.35
    # print "c", 2 * c**2
    # return np.exp(-1 * np.power(x, 2.) / (2 * c**2))

    def get_utility(self, samples):
        utility = np.dot(self.conf,
                         np.max(self.gauss_val_masked[:, samples], axis=1))
        return utility

    def get_argmax_utility(self, samples, others, prior):
        sample_gauss = np.zeros((self.num_inputs, 1))
        if len(samples) > 0:
            sample_gauss = np.max(
                self.gauss_val_masked[:, samples], axis=1).reshape(
                    (self.num_inputs, 1))
        utility_array = np.sum(
            self.conf.reshape(
                (self.num_inputs,
                 1)) * np.maximum(sample_gauss, self.gauss_val[:, others]),
            axis=0)
        if prior is not None:
            utility_array = utility_array * prior[others]
        argmax = np.argmax(utility_array)
        return utility_array[argmax], argmax

    # def get_mean_utility(self, samples, others):
    # sample_gauss = np.zeros((self.num_inputs, 1))
    # if len(samples) > 0:
    # sample_gauss = np.max(
    # self.gauss_val_masked[:, samples], axis=1).reshape(
    # (self.num_inputs, 1))
    # mean_utility = np.mean(
    # np.sum(
    # self.conf.reshape((self.num_inputs, 1)) *
    # np.maximum(sample_gauss, self.gauss_val_masked[:, others]),
    # axis=0))
    # return mean_utility

    # def get_mean_max_possible_utility(self, samples, others):
    # sample_gauss = np.zeros((self.num_inputs, 1))
    # if len(samples) > 0:
    # sample_gauss = np.max(
    # self.gauss_val_masked[:, samples], axis=1).reshape(
    # (self.num_inputs, 1))
    # utility = np.sum(
    # np.max(self.conf, axis=1) * np.mean(
    # np.maximum(sample_gauss, self.gauss_val[:, others]), axis=1))
    # return utility

    def _plot_coverage_conf(self, outfile, outfile2):
        cands = self.get_cand_unkunks()
        gauss_sum = np.sum(
            self.conf.reshape((self.num_inputs, 1)) * self.gauss_val[:, cands],
            axis=0)
        confs = self.conf[cands]
        colors = []
        for i in cands:
            if self._is_unkunk(i):
                colors.append("r")
            else:
                colors.append("b")
        plt.scatter(confs, gauss_sum, color=colors)
        plt.savefig(outfile)
        plt.ylabel("Coverage")
        plt.xlabel("Conf")
        plt.close()

        plt.scatter(confs, gauss_sum * (1 - confs), color=colors)
        plt.savefig(outfile2)
        plt.ylabel("Coverage x (1-conf)")
        plt.xlabel("Conf")
        plt.close()

    def save(self, outfile):
        np.savez(
            outfile,
            unkunks=self.unkunks,
            distance_matrix=self.distance_matrix,
            gauss_val=self.gauss_val,
            gauss_val_masked=self.gauss_val_masked)
