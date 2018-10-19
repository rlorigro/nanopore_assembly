import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from handlers.FileManager import FileManager
from tqdm import tqdm
from collections import Counter
from itertools import product
from math import lgamma, exp, log
from sys import float_info
import pickle
import csv


# JOINT_DISTRIBUTION_PATH = "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/distribution/distribution_2018_10_3_8_21.pkl"
JOINT_DISTRIBUTION_PATH = "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/distribution/distribution_2018_10_16_10_23_38_123997.pkl"    # transition anchored windows


class JointClassifier:
    def __init__(self, distribution_path, max_runlength=50):
        self.distribution = self.load_pickled_distribution(distribution_path)
        self.max_runlength = max_runlength

        self.log_conditional_memo = self.make_log_conditional_memo(self.distribution)

    def complement(self, base):
        if base == "A":
            return "T"
        elif base == "C":
            return "G"
        elif base == "G":
            return "C"
        elif base == "T":
            return "A"
        else:
            return "-"

    def make_log_conditional_memo(self, distr, max_run_length=50):
        rlbs = [('-', 0)] + list(product("ACGT", range(1, max_run_length + 1)))
        memo = {}
        for called_rlb in rlbs:
            normalizing_factor = 0.0
            for true_rlb in rlbs:
                normalizing_factor += distr[(called_rlb, true_rlb)]
            for true_rlb in rlbs:
                memo[(called_rlb, true_rlb)] = log(distr[(called_rlb, true_rlb)] / normalizing_factor)

        return memo

    def get_consensus_posterior(self, pileup):
        """
        Pileup should be a list of (base, length, strand) tuples, where strand == True indicates the reverse strand
        :param pileup:
        :return:
        """
        posterior = {}
        max_log_likelihood = -float_info.max
        rlbs = [('-', 0)] + list(product("ACGT", range(1, self.max_runlength + 1)))

        for true_rlb in rlbs:
            ll = 0.0

            for base, length, strand in pileup:
                if strand:
                    called = self.complement(base)
                    true = self.complement(true_rlb[0])
                else:
                    called = base
                    true = true_rlb[0]

                ll += self.log_conditional_memo[((called, length), (true, true_rlb[1]))]

            posterior[true_rlb] = ll
            max_log_likelihood = max(ll, max_log_likelihood)

        for true_rlb in rlbs:
            posterior[true_rlb] = exp(posterior[true_rlb] - max_log_likelihood)

        total = sum(posterior.values())

        max_posterior = -sys.maxsize
        max_prediction = None
        for true_rlb in rlbs:
            posterior[true_rlb] /= total

            if posterior[true_rlb] > max_posterior:
                max_posterior = posterior[true_rlb]
                max_prediction = true_rlb

        return posterior, max_posterior, max_prediction

    def load_pickled_distribution(self, path):
        with open(path, 'rb') as pickle_file:
            distribution = pickle.load(pickle_file)

        return distribution


def test():
    # max_runlength = 50
    joint_classifier = JointClassifier(JOINT_DISTRIBUTION_PATH)

    # print(type(distribution))
    # print(distribution.keys())
    #
    # for key in distribution:
    #     base = key[0]
    #     runlength = key[1]

    pileup_columns = [[('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('G', 3, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False)],
        [('T', 4, True), ('T', 4, True), ('T', 4, False), ('T', 4, False), ('T', 4, False), ('T', 4, True),
         ('T', 4, True),
         ('T', 4, True), ('T', 4, True), ('T', 4, False), ('T', 4, True), ('T', 4, True), ('T', 4, False),
         ('T', 4, False),
         ('T', 4, True), ('T', 3, False), ('T', 1, False), ('T', 4, True), ('T', 4, False), ('T', 4, False),
         ('T', 4, True),
         ('T', 4, True), ('T', 4, True), ('T', 4, False), ('T', 4, False), ('T', 4, True), ('T', 4, True),
         ('T', 4, False),
         ('T', 5, False), ('T', 4, True), ('T', 4, True), ('T', 4, True), ('T', 4, False), ('T', 4, False),
         ('T', 4, False),
         ('T', 4, False), ('T', 4, True), ('T', 4, False), ('T', 4, True), ('T', 4, True), ('T', 4, True),
         ('T', 4, True),
         ('T', 4, True)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('A', 2, False), ('A', 4, False), ('-', 0, False), ('A', 4, False), ('A', 1, False), ('-', 0, False),
         ('-', 0, False), ('A', 3, False), ('A', 1, True), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('A', 4, False), ('A', 3, False), ('-', 0, False),
         ('-', 0, False), ('A', 4, True), ('A', 4, False), ('A', 1, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('A', 4, True)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('C', 1, True), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('T', 3, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('T', 3, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('T', 3, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('T', 1, True), ('T', 3, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('T', 3, True)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('G', 1, False), ('G', 1, False), ('-', 0, False), ('G', 1, False), ('G', 1, False), ('-', 0, False),
         ('-', 0, False), ('G', 1, False), ('G', 1, True), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('G', 1, False), ('G', 1, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('G', 1, False), ('G', 1, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('G', 1, True)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('T', 1, True), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('T', 1, True)],
        [('A', 4, True), ('A', 4, True), ('A', 3, False), ('A', 4, False), ('A', 4, False), ('A', 4, True),
         ('A', 3, True),
         ('A', 5, True), ('A', 4, True), ('A', 4, False), ('A', 5, True), ('A', 4, True), ('A', 1, False),
         ('A', 1, False),
         ('A', 5, True), ('A', 2, False), ('A', 2, False), ('A', 4, True), ('A', 4, False), ('A', 9, False),
         ('A', 2, True),
         ('A', 4, True), ('A', 4, True), ('A', 4, False), ('A', 4, False), ('A', 3, True), ('A', 4, True),
         ('A', 6, False),
         ('A', 2, False), ('A', 4, True), ('A', 4, True), ('-', 0, False), ('A', 8, False), ('A', 1, False),
         ('A', 4, False), ('A', 4, False), ('A', 4, True), ('A', 4, False), ('A', 4, True), ('A', 4, True),
         ('A', 4, True),
         ('A', 4, True), ('A', 4, True)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('G', 1, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('C', 1, True), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('C', 1, True), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False)],
        [('T', 3, True), ('T', 3, True), ('T', 3, False), ('T', 3, False), ('T', 3, False), ('T', 3, True),
         ('T', 4, True),
         ('T', 3, True), ('T', 3, True), ('T', 3, False), ('T', 3, True), ('T', 3, True), ('T', 3, False),
         ('-', 0, False),
         ('T', 3, True), ('T', 3, False), ('T', 3, False), ('T', 3, True), ('T', 3, False), ('T', 1, False),
         ('T', 2, True),
         ('T', 3, True), ('T', 3, True), ('T', 3, False), ('T', 3, False), ('T', 2, True), ('T', 3, True),
         ('T', 1, False),
         ('T', 3, False), ('T', 3, True), ('T', 3, True), ('T', 1, True), ('T', 1, False), ('T', 3, False),
         ('T', 3, False),
         ('T', 3, False), ('T', 3, True), ('T', 3, False), ('T', 3, True), ('T', 3, True), ('T', 3, True),
         ('T', 3, True),
         ('-', 0, False)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('A', 1, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False)],
        [('G', 1, True), ('G', 1, True), ('G', 1, False), ('G', 1, False), ('G', 1, False), ('G', 1, True),
         ('G', 1, True),
         ('G', 1, True), ('G', 1, True), ('G', 1, False), ('G', 1, True), ('G', 1, True), ('G', 1, False),
         ('G', 1, False),
         ('G', 1, True), ('G', 1, False), ('G', 1, False), ('G', 1, True), ('G', 1, False), ('G', 1, False),
         ('G', 1, True),
         ('G', 1, True), ('G', 1, True), ('G', 1, False), ('G', 1, False), ('G', 1, True), ('G', 1, True),
         ('G', 1, False),
         ('G', 1, False), ('G', 1, True), ('G', 1, True), ('G', 1, True), ('G', 1, False), ('G', 1, False),
         ('G', 1, False),
         ('G', 1, False), ('G', 1, True), ('G', 1, False), ('G', 1, True), ('G', 1, True), ('G', 1, True),
         ('G', 1, True),
         ('-', 0, False)],
        [('A', 5, True), ('A', 5, True), ('A', 13, False), ('A', 9, False), ('A', 8, False), ('A', 9, True),
         ('A', 5, True),
         ('A', 4, True), ('A', 4, True), ('A', 5, False), ('A', 4, True), ('A', 8, True), ('A', 11, False),
         ('A', 8, False),
         ('A', 7, True), ('A', 8, False), ('A', 8, False), ('A', 5, True), ('A', 9, False), ('-', 0, False),
         ('A', 6, True),
         ('A', 4, True), ('A', 7, True), ('A', 5, False), ('A', 7, False), ('A', 5, True), ('A', 8, True),
         ('-', 0, False),
         ('A', 10, False), ('A', 3, True), ('A', 6, True), ('A', 4, True), ('-', 0, False), ('A', 7, False),
         ('A', 7, False), ('A', 6, False), ('A', 6, True), ('A', 8, False), ('A', 8, True), ('A', 6, True),
         ('A', 5, True),
         ('A', 7, True), ('-', 0, False)],
        [('T', 1, True), ('T', 1, True), ('T', 1, False), ('T', 1, False), ('T', 1, False), ('T', 1, True),
         ('T', 1, True),
         ('-', 0, False), ('T', 1, True), ('T', 1, False), ('T', 1, True), ('-', 0, False), ('T', 1, False),
         ('T', 1, False), ('T', 1, True), ('T', 1, False), ('T', 1, False), ('T', 1, True), ('T', 1, False),
         ('T', 2, False), ('T', 1, True), ('T', 1, True), ('T', 1, True), ('T', 1, False), ('T', 1, False),
         ('T', 1, True),
         ('T', 1, True), ('T', 2, False), ('T', 1, False), ('-', 0, False), ('T', 1, True), ('T', 1, True),
         ('T', 2, False),
         ('T', 1, False), ('T', 1, False), ('T', 1, False), ('T', 1, True), ('T', 1, False), ('T', 1, True),
         ('T', 1, True),
         ('T', 1, True), ('T', 1, True), ('T', 1, True)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('A', 2, True), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('A', 3, True), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False)],
        [('G', 1, True), ('G', 1, True), ('G', 1, False), ('A', 2, False), ('G', 1, False), ('G', 1, True),
         ('G', 1, True),
         ('-', 0, False), ('G', 1, True), ('G', 1, False), ('G', 1, True), ('-', 0, False), ('G', 1, False),
         ('G', 1, False), ('G', 1, True), ('G', 2, False), ('G', 1, False), ('G', 1, True), ('G', 1, False),
         ('G', 2, False), ('G', 1, True), ('G', 1, True), ('G', 1, True), ('G', 1, False), ('G', 1, False),
         ('G', 1, True),
         ('G', 1, True), ('G', 1, False), ('G', 1, False), ('-', 0, False), ('G', 1, True), ('G', 1, True),
         ('G', 1, False),
         ('A', 1, False), ('G', 1, False), ('G', 1, False), ('G', 1, True), ('A', 1, False), ('G', 1, True),
         ('G', 1, True),
         ('G', 1, True), ('G', 1, True), ('G', 1, True)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('C', 1, True),
         ('C', 1, True), ('C', 1, True), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('C', 1, True), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('C', 1, True)],
        [('T', 2, True), ('T', 2, True), ('T', 2, False), ('T', 3, False), ('T', 2, False), ('T', 1, True),
         ('-', 0, False),
         ('-', 0, False), ('T', 2, True), ('T', 2, False), ('T', 2, True), ('T', 1, True), ('T', 2, False),
         ('T', 2, False),
         ('T', 2, True), ('T', 1, False), ('T', 2, False), ('T', 2, True), ('T', 1, False), ('-', 0, False),
         ('T', 1, True),
         ('T', 2, True), ('T', 2, True), ('T', 2, False), ('T', 2, False), ('T', 2, True), ('T', 2, True),
         ('-', 0, False),
         ('T', 2, False), ('T', 1, True), ('T', 2, True), ('T', 2, True), ('-', 0, False), ('T', 3, False),
         ('T', 2, False),
         ('T', 2, False), ('T', 2, True), ('T', 3, False), ('T', 2, True), ('T', 2, True), ('T', 2, True),
         ('T', 2, True),
         ('T', 1, True)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('G', 2, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False)],
        [('-', 0, False), ('-', 0, False), ('A', 1, False), ('-', 0, False), ('A', 1, False), ('-', 0, False),
         ('-', 0, False), ('A', 1, True), ('A', 1, True), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('A', 1, False), ('-', 0, False), ('A', 1, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('A', 1, False), ('A', 1, True), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('A', 1, False), ('A', 1, False), ('A', 2, True),
         ('-', 0, False), ('-', 0, False), ('A', 1, False), ('A', 1, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('A', 1, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False)],
        [('G', 2, True), ('G', 3, True), ('G', 2, False), ('G', 3, False), ('G', 2, False), ('G', 3, True),
         ('G', 3, True),
         ('G', 1, True), ('-', 0, False), ('G', 3, False), ('G', 2, True), ('G', 1, True), ('G', 3, False),
         ('G', 2, False),
         ('G', 3, True), ('G', 2, False), ('G', 3, False), ('G', 3, True), ('G', 3, False), ('G', 1, False),
         ('G', 1, True),
         ('G', 2, True), ('G', 3, True), ('G', 3, False), ('G', 3, False), ('G', 3, True), ('G', 3, True),
         ('G', 1, False),
         ('G', 1, False), ('G', 2, True), ('G', 3, True), ('G', 1, True), ('G', 1, False), ('G', 2, False),
         ('G', 3, False),
         ('G', 3, False), ('G', 3, True), ('G', 2, False), ('G', 3, True), ('G', 3, True), ('G', 1, True),
         ('G', 3, True),
         ('G', 3, True)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('T', 1, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('T', 1, False), ('T', 1, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('T', 1, True), ('-', 0, False),
         ('-', 0, False)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('C', 1, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('C', 1, False), ('C', 1, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('C', 1, True), ('-', 0, False),
         ('-', 0, False)],
        [('T', 1, True), ('T', 1, True), ('T', 1, False), ('T', 1, False), ('T', 1, False), ('T', 1, True),
         ('-', 0, False),
         ('T', 1, True), ('T', 2, True), ('T', 1, False), ('T', 1, True), ('-', 0, False), ('T', 1, False),
         ('T', 1, False),
         ('T', 1, True), ('T', 1, False), ('T', 1, False), ('T', 1, True), ('T', 1, False), ('T', 1, False),
         ('T', 1, True),
         ('T', 1, True), ('T', 1, True), ('T', 1, False), ('T', 1, False), ('-', 0, False), ('T', 1, True),
         ('-', 0, False),
         ('T', 1, False), ('-', 0, False), ('T', 1, True), ('-', 0, False), ('-', 0, False), ('T', 1, False),
         ('T', 1, False), ('T', 1, False), ('T', 1, True), ('T', 1, False), ('T', 1, True), ('T', 1, True),
         ('T', 1, True),
         ('T', 1, True), ('T', 1, True)],
        [('C', 2, True), ('C', 1, True), ('C', 1, False), ('C', 1, False), ('C', 1, False), ('C', 1, True),
         ('C', 1, True),
         ('C', 1, True), ('C', 2, True), ('C', 1, False), ('C', 1, True), ('-', 0, False), ('C', 1, False),
         ('C', 2, False),
         ('C', 1, True), ('C', 1, False), ('C', 1, False), ('C', 1, True), ('C', 1, False), ('C', 1, False),
         ('C', 1, True),
         ('C', 1, True), ('C', 1, True), ('C', 1, False), ('C', 1, False), ('-', 0, False), ('C', 1, True),
         ('-', 0, False),
         ('C', 1, False), ('-', 0, False), ('C', 1, True), ('-', 0, False), ('-', 0, False), ('C', 1, False),
         ('C', 1, False), ('C', 1, False), ('C', 1, True), ('C', 1, False), ('C', 1, True), ('C', 1, True),
         ('C', 1, True),
         ('C', 1, True), ('C', 1, True)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('A', 1, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('A', 2, True), ('-', 0, False),
         ('-', 0, False)],
        [('T', 3, True), ('T', 1, True), ('T', 1, False), ('T', 1, False), ('T', 1, False), ('T', 1, True),
         ('T', 2, True),
         ('T', 1, True), ('-', 0, False), ('T', 1, False), ('T', 1, True), ('T', 2, True), ('T', 1, False),
         ('T', 1, False),
         ('T', 1, True), ('T', 1, False), ('T', 1, False), ('T', 1, True), ('T', 1, False), ('T', 1, False),
         ('T', 1, True),
         ('T', 1, True), ('T', 1, True), ('T', 1, False), ('T', 1, False), ('T', 1, True), ('T', 1, True),
         ('T', 1, False),
         ('T', 1, False), ('T', 1, True), ('T', 1, True), ('T', 1, True), ('T', 1, False), ('T', 1, False),
         ('T', 2, False),
         ('T', 1, False), ('T', 1, True), ('T', 1, False), ('T', 1, True), ('T', 1, True), ('-', 0, False),
         ('T', 1, True),
         ('T', 1, True)],
        [('-', 0, False), ('C', 1, True), ('-', 0, False), ('C', 1, False), ('-', 0, False), ('C', 1, True),
         ('C', 1, True),
         ('-', 0, False), ('-', 0, False), ('C', 1, False), ('C', 1, True), ('-', 0, False), ('-', 0, False),
         ('C', 1, False), ('-', 0, False), ('C', 1, False), ('-', 0, False), ('C', 1, True), ('-', 0, False),
         ('C', 1, False), ('C', 1, True), ('C', 1, True), ('C', 1, True), ('C', 1, False), ('-', 0, False),
         ('C', 3, True),
         ('C', 1, True), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('C', 1, True), ('C', 2, True),
         ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('C', 1, True), ('-', 0, False), ('C', 1, True),
         ('C', 1, True),
         ('-', 0, False), ('C', 1, True), ('C', 1, True)],
        [('G', 1, True), ('G', 1, True), ('-', 0, False), ('G', 1, False), ('-', 0, False), ('G', 1, True),
         ('G', 1, True),
         ('-', 0, False), ('G', 1, True), ('G', 1, False), ('G', 1, True), ('G', 1, True), ('-', 0, False),
         ('G', 1, False),
         ('G', 1, True), ('G', 1, False), ('-', 0, False), ('G', 1, True), ('-', 0, False), ('G', 1, False),
         ('G', 1, True),
         ('G', 1, True), ('G', 1, True), ('G', 1, False), ('-', 0, False), ('G', 1, True), ('G', 1, True),
         ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('G', 1, True), ('G', 2, True), ('-', 0, False), ('G', 1, False),
         ('-', 0, False), ('-', 0, False), ('G', 1, True), ('-', 0, False), ('G', 1, True), ('G', 1, True),
         ('G', 1, True),
         ('G', 1, True), ('G', 1, True)],
        [('-', 0, False), ('C', 2, True), ('C', 1, False), ('-', 0, False), ('C', 1, False), ('-', 0, False),
         ('-', 0, False), ('C', 1, True), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('C', 1, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('C', 1, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('C', 1, False), ('-', 0, False), ('-', 0, False), ('C', 1, False), ('C', 1, False), ('C', 1, True),
         ('-', 0, False), ('-', 0, False), ('C', 2, False), ('-', 0, False), ('-', 0, False), ('C', 1, False),
         ('-', 0, False), ('C', 1, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False)],
        [('-', 0, False), ('A', 1, True), ('A', 1, False), ('-', 0, False), ('A', 1, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('A', 1, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('A', 1, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('A', 1, False), ('A', 1, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('A', 1, False),
         ('-', 0, False), ('A', 1, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False)],
        [('C', 2, True), ('C', 1, True), ('T', 1, False), ('C', 2, False), ('T', 1, False), ('C', 2, True),
         ('C', 2, True),
         ('T', 1, True), ('C', 2, True), ('C', 1, False), ('C', 2, True), ('-', 0, False), ('T', 1, False),
         ('C', 2, False),
         ('C', 2, True), ('C', 2, False), ('C', 1, False), ('C', 2, True), ('T', 1, False), ('C', 2, False),
         ('C', 2, True),
         ('C', 2, True), ('C', 2, True), ('C', 2, False), ('T', 2, False), ('C', 2, True), ('C', 2, True),
         ('T', 1, False),
         ('T', 1, False), ('T', 1, True), ('C', 2, True), ('C', 1, True), ('T', 1, False), ('C', 1, False),
         ('C', 1, False),
         ('T', 1, False), ('C', 2, True), ('T', 1, False), ('C', 2, True), ('C', 2, True), ('C', 1, True),
         ('C', 2, True),
         ('C', 2, True)],
        [('A', 1, True), ('A', 1, True), ('A', 1, False), ('A', 1, False), ('-', 0, False), ('A', 1, True),
         ('A', 1, True),
         ('A', 1, True), ('A', 1, True), ('A', 1, False), ('A', 1, True), ('A', 1, True), ('A', 3, False),
         ('A', 1, False),
         ('A', 1, True), ('A', 1, False), ('A', 2, False), ('A', 1, True), ('A', 1, False), ('A', 1, False),
         ('A', 2, True),
         ('A', 1, True), ('A', 1, True), ('A', 1, False), ('A', 1, False), ('A', 1, True), ('A', 1, True),
         ('A', 1, False),
         ('A', 1, False), ('A', 1, True), ('A', 1, True), ('A', 2, True), ('-', 0, False), ('A', 1, False),
         ('A', 1, False),
         ('A', 1, False), ('A', 1, True), ('A', 1, False), ('A', 1, True), ('A', 1, True), ('A', 1, True),
         ('A', 1, True),
         ('A', 1, True)],
        [('C', 1, True), ('C', 1, True), ('C', 1, False), ('C', 1, False), ('-', 0, False), ('C', 1, True),
         ('C', 1, True),
         ('T', 3, True), ('C', 1, True), ('C', 1, False), ('C', 1, True), ('T', 1, True), ('-', 0, False),
         ('C', 1, False),
         ('C', 1, True), ('C', 1, False), ('C', 1, False), ('C', 1, True), ('C', 1, False), ('C', 1, False),
         ('T', 1, True),
         ('C', 1, True), ('C', 1, True), ('C', 1, False), ('-', 0, False), ('T', 1, True), ('C', 1, True),
         ('C', 1, False),
         ('C', 1, False), ('C', 1, True), ('C', 1, True), ('T', 1, True), ('C', 3, False), ('-', 0, False),
         ('C', 1, False),
         ('C', 1, False), ('C', 1, True), ('C', 1, False), ('C', 1, True), ('C', 1, True), ('C', 1, True),
         ('C', 1, True),
         ('C', 1, True)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('T', 1, True),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False)],
        [('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('A', 1, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False), ('-', 0, False),
         ('-', 0, False)],
        [('G', 1, True), ('G', 1, True), ('G', 1, False), ('G', 1, False), ('-', 0, False), ('G', 1, True),
         ('G', 1, True),
         ('G', 1, True), ('G', 1, True), ('G', 1, False), ('G', 1, True), ('G', 1, True), ('G', 1, False),
         ('G', 1, False),
         ('G', 1, True), ('G', 1, False), ('-', 0, False), ('G', 1, True), ('G', 1, False), ('G', 1, False),
         ('G', 1, True),
         ('G', 1, True), ('G', 1, True), ('G', 1, False), ('G', 1, False), ('G', 1, True), ('G', 1, True),
         ('G', 1, False),
         ('G', 1, False), ('G', 1, True), ('G', 1, True), ('G', 1, True), ('G', 1, False), ('G', 1, False),
         ('-', 0, False),
         ('G', 1, False), ('G', 1, True), ('-', 0, False), ('G', 1, True), ('G', 1, True), ('G', 1, True),
         ('G', 1, True),
         ('G', 1, True)],
        [('A', 2, True), ('A', 3, True), ('A', 3, False), ('A', 2, False), ('A', 4, False), ('A', 3, True),
         ('A', 1, True),
         ('A', 4, True), ('A', 3, True), ('A', 3, False), ('A', 4, True), ('A', 3, True), ('A', 2, False),
         ('A', 3, False),
         ('A', 3, True), ('A', 3, False), ('A', 3, False), ('A', 3, True), ('A', 3, False), ('A', 2, False),
         ('A', 3, True),
         ('A', 3, True), ('A', 3, True), ('A', 2, False), ('A', 3, False), ('A', 3, True), ('A', 3, True),
         ('A', 2, False),
         ('A', 2, False), ('A', 3, True), ('A', 3, True), ('A', 4, True), ('A', 3, False), ('A', 2, False),
         ('A', 3, False),
         ('A', 2, False), ('A', 3, True), ('A', 3, False), ('A', 5, True), ('A', 3, True), ('A', 3, True),
         ('A', 5, True),
         ('A', 3, True)]]

    # TTTTAAAATTTG---AAAAATGTT--GTCT-G-CACGAAA
    # TTTTAAAATTTGAAAAAAAATGTTGGGTCTCGCCACGAAA

    for pileup in pileup_columns:
        posterior, max_posterior, max_prediction = joint_classifier.get_consensus_posterior(pileup=pileup)
        print(max_prediction, max_posterior)

    # for rlb in sorted(posterior.keys(), key=lambda x: posterior[x], reverse=True):
    #     print("{0}: {1}".format(rlb, posterior[rlb]))


if __name__ == "__main__":
    test()



