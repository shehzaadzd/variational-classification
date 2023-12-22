## This file computes the reliability histogram and computes the ECE and KS distance of the given system

import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import os


class Evaluator(object):
    def __init__(self,):
        pass


    @staticmethod
    def sec_classification(y_pred, conf, save=False, save_name=None):
        """Compute the AURC.
        Args:
          y_true: true labels, vector of size n_test
          y_pred: predicted labels by the classifier, vector of size n_test
          conf: confidence associated to y_pred, vector of size n_test
        Returns:
          conf: confidence sorted (in decreasing order)
          risk_cov: risk vs coverage (increasing coverage from 0 to 1)
          aurc: AURC
          eaurc: Excess AURC
        """
        y_pred = np.array(y_pred)
        conf = np.array(conf)

        y_true = np.ones_like(y_pred)
        n = len(y_true)
        # ninety = np.ceil(0.9*n)
        # eighty = np.ceil(0.8*n)
        ind = np.argsort(conf)
        y_true, y_pred, conf = y_true[ind][::-1], y_pred[ind][::-1], conf[ind][::-1]
        risk_cov = np.divide(np.cumsum(y_true != y_pred).astype(np.float),
                             np.arange(1, n + 1))
        if save and save_name:
            np.save("{}.npy".format(save_name), risk_cov)
        nrisk = np.sum(y_true != y_pred)
        aurc = np.mean(risk_cov)
        opt_aurc = (1. / n) * np.sum(np.divide(np.arange(1, nrisk + 1).astype(np.float),
                                               n - nrisk + np.arange(1, nrisk + 1)))
        coverage_90 = (np.divide(np.cumsum(y_true == y_pred).astype(np.float),
                                 np.arange(1, n + 1)) > 0.9).astype(np.float).sum() / n
        best_c_90 = (np.divide(np.cumsum(np.sort(y_pred)[::-1]).astype(np.float),
                               np.arange(1, n + 1)) > 0.9).astype(np.float).sum() / n
        coverage_80 = (np.divide(np.cumsum(y_true == y_pred).astype(np.float),
                                 np.arange(1, n + 1)) > 0.8).astype(np.float).sum() / n
        best_c_80 = (np.divide(np.cumsum(np.sort(y_pred)[::-1]).astype(np.float),
                               np.arange(1, n + 1)) > 0.9).astype(np.float).sum() / n

        eaurc = aurc - opt_aurc

        return risk_cov, aurc, opt_aurc, eaurc, coverage_80, coverage_90, best_c_80, best_c_90


    @staticmethod
    def create_bins(acc, conf, bins="normalized"):
        '''

        :param acc:
        :param conf:
        :param bins:
        :return:
        '''

        assert len(acc) == len(conf)
        if bins == "normalized":
            bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        if bins == "equal_width":
            hist, bins = np.histogram(conf)
        which_bin = np.digitize(conf, bins)
        bin_accuracy = np.zeros(len(bins) - 1)
        bin_len = np.zeros_like(bin_accuracy)
        bin_mean_confidence = np.zeros_like(bin_accuracy)
        n = len(acc)
        ECE = 0.0
        MCE = 0.0


        #calculating bin accuracies
        for c, a, b_id_ in zip(conf, acc, which_bin):
            b_id = b_id_ - 1
            if b_id_ == len(bins):
                b_id = b_id_ -2
            bin_accuracy[b_id] += a
            bin_mean_confidence[b_id] += c
            bin_len[b_id] += 1


        for i, l in enumerate(bin_len):
            if l > 0:
                bin_accuracy[i] /= l
                bin_mean_confidence[i] /= l


        for a, c, l in zip(bin_accuracy, bin_mean_confidence, bin_len):
            if l > 0:
                CE = np.absolute(a - c) * l
                ECE += CE
                if  CE > MCE:
                    MCE = CE
            else:
                ECE += 0
        ECE /= n
        MCE /= n

        return ECE, MCE, bin_accuracy, bin_mean_confidence, bin_len

    @staticmethod
    def create_fine_bins(acc, conf, bins="normalized"):
        '''

        :param acc:
        :param conf:
        :param bins:
        :return:
        '''

        assert len(acc) == len(conf)
        if bins == "normalized":
            bins = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95, 1.0]
        if bins == "equal_width":
            hist, bins = np.histogram(conf)
        which_bin = np.digitize(conf, bins)
        bin_accuracy = np.zeros(len(bins) - 1)
        bin_len = np.zeros_like(bin_accuracy)
        bin_mean_confidence = np.zeros_like(bin_accuracy)
        n = len(acc)
        ECE = 0.0
        MCE = 0.0


        #calculating bin accuracies
        for c, a, b_id_ in zip(conf, acc, which_bin):
            b_id = b_id_ - 1
            if b_id_ == len(bins):
                b_id = b_id_ -2
            bin_accuracy[b_id] += a
            bin_mean_confidence[b_id] += c
            bin_len[b_id] += 1


        for i, l in enumerate(bin_len):
            if l > 0:
                bin_accuracy[i] /= l
                bin_mean_confidence[i] /= l


        for a, c, l in zip(bin_accuracy, bin_mean_confidence, bin_len):
            if l > 0:
                CE = np.absolute(a - c) * l
                ECE += CE
                if  CE > MCE:
                    MCE = CE
            else:
                ECE += 0
        ECE /= n
        MCE /= n

        return ECE, MCE, bin_accuracy, bin_mean_confidence, bin_len

    @staticmethod
    def continuous_ece(acc, conf):

        confacc = [(c, a) for c, a in zip(conf, acc)]
        sorted_confacc = sorted(confacc, key = lambda  x: x[0])

        newks = 0
        for c, a in sorted_confacc:
            newks += np.absolute(c - a )

        return  newks/ len(confacc)




    @staticmethod
    def KS(acc, conf):
        sorted_acc = np.sort(acc, )
        sorted_conf = np.sort(conf)

        sum_acc = 0
        sum_conf = 0
        max_diff = 0
        for a, c in zip(sorted_acc, sorted_conf):
            sum_acc += a
            sum_conf += c
            diff = np.absolute(sum_acc - sum_conf)
            if diff > max_diff:
                max_diff = diff

        return max_diff/len(acc)

    @staticmethod
    def draw_plot(binned_accuracy, binned_confidence):
        plt.style.use('ggplot')

        bin_size = 0.1
        positions = np.arange(0+bin_size/2, 1+bin_size/2, bin_size)
        plt.bar( positions, binned_confidence ,width = 0.1, edgecolor = "red", color = "red", alpha = 0.3, label="Gap", linewidth=2, zorder=2 )

        plt.bar( positions, binned_accuracy, width = 0.1, edgecolor = "black", color = "blue", label="Outputs", zorder = 3 )

        plt.xlabel("confidence")
        plt.ylabel("accuracy")
        ax = plt.axes()
        ax.set_aspect('equal')

        ax.plot([0,1], [0,1], linestyle = "--")
        # ax.legend(handles = [gap_plt, output_plt])
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        plt.figure(0)

    @staticmethod
    def save_plot(binned_accuracy, binned_confidence, name):
        plt.style.use('ggplot')

        bin_size = 0.1
        positions = np.arange(0+bin_size/2, 1+bin_size/2, bin_size)
        plt.bar( positions, binned_confidence ,width = 0.1, edgecolor = "red", color = "red", alpha = 0.3, label="Gap", linewidth=2, zorder=2 )

        plt.bar( positions, binned_accuracy, width = 0.1, edgecolor = "black", color = "blue", label="Outputs", zorder = 3, alpha=0.6 )

        plt.xlabel("confidence")
        plt.ylabel("accuracy")
        ax = plt.axes()
        ax.set_aspect('equal')

        ax.plot([0,1], [0,1], linestyle = "--")
        # ax.legend(handles = [gap_plt, output_plt])
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        plt.savefig("../Dropbox/graphs/{}.pdf".format(name))

    @staticmethod
    def save_plot_fine(binned_accuracy, binned_confidence, name):
        plt.style.use('ggplot')

        bin_size = 0.05
        positions = np.arange(0+bin_size/2, 1+bin_size/2, bin_size)
        plt.bar( positions, binned_confidence ,width = 0.05, edgecolor = "red", color = "red", alpha = 0.3, label="Gap", linewidth=2, zorder=2 )

        plt.bar( positions, binned_accuracy, width = 0.05, edgecolor = "black", color = "blue", label="Outputs", zorder = 3, alpha=0.6 )

        plt.xlabel("confidence")
        plt.ylabel("accuracy")
        ax = plt.axes()
        ax.set_aspect('equal')

        ax.plot([0,1], [0,1], linestyle = "--")
        # ax.legend(handles = [gap_plt, output_plt])
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        # os.remove("graphs/{}.pdf".format(name))
        plt.savefig("graphs/{}.pdf".format(name))