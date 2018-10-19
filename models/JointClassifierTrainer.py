# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 11:05:45 2018
(primary) Author: Jordan Eizenga
"""

import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from handlers.FileManager import FileManager
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
from collections import Counter
from itertools import product
from math import lgamma, exp, log
from sys import float_info
import pickle
import csv

# data should be given as a list of tuples, where each tuple contains
# two tuples, each of which contains the base and its run length
# the first of the two tuples should be the called run length base, and 
# the second should the true run length base
# e.g. [(('A', 2),('A',4)), (('G', 7),('G', 2))]
# a run of 0 bases should be encoded ('-', 0)


class JointClassifierTrainer:
    def __init__(self):
        self.counts = None

    @staticmethod
    def count_joint_frequency(p, path, return_dict):
        with open(path, 'rb') as pickle_file:
            tuples = pickle.load(pickle_file)
            counts = Counter(tuples)

        return_dict[p] = counts

        # print(p)

    def get_counts_from_tuples(self, paths, max_threads=24):
        print("Paths loaded: ", len(paths))

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        print("counting tuples...")

        args = list()
        for p, path in enumerate(paths):
            args.append([p, path, return_dict])

        # args = args[:30] # debug

        n_threads = min(len(args), max_threads)

        with Pool(processes=n_threads) as pool:
            pool.starmap(self.count_joint_frequency, args)

        print("counted all files...")

        total = Counter()
        for key in return_dict.keys():
            total = total + return_dict[key]

        return total

    # def train_from_pileups(self, path):

    def train_model(self, counts, max_run_length=50, pseudocounts=1, correct_call_pseudocount_bonus=2):
        # list of all possible run length bases up to the max length
        rlbs = [('-', 0)] + list(product("ACGT", range(1, max_run_length + 1)))

        # where we record the probabilities
        distr = {}

        # sum of alphas
        total_pseudocount = len(rlbs) * len(rlbs) * pseudocounts + len(rlbs) * correct_call_pseudocount_bonus

        # sum of tilde alphas
        total_posterior_param = total_pseudocount + sum(counts.values())

        for call_true_pair in product(rlbs, repeat=2):
            pair_pseudocounts = pseudocounts
            if call_true_pair[0] == call_true_pair[1]:
                pair_pseudocounts += correct_call_pseudocount_bonus

            # tilde alpha
            posterior_param = pair_pseudocounts + counts[call_true_pair]

            # equation (5) in write up
            log_prob = lgamma(total_posterior_param) + lgamma(1.0 + posterior_param) \
                       - lgamma(1.0 + total_posterior_param) - lgamma(posterior_param)

            distr[call_true_pair] = exp(log_prob)

        return distr


def train_model(data, max_run_length=50, pseudocounts=1, correct_call_pseudocount_bonus=2):
    # the data can be summarized by the count of each call/true pair of run length bases
    count = Counter(data)

    # list of all possible run length bases up to the max length
    rlbs = [('-', 0)] + list(product("ACGT", range(1, max_run_length + 1)))

    # where we record the probabilities
    distr = {}

    # sum of alphas
    total_pseudocount = len(rlbs) * len(rlbs) * pseudocounts + len(rlbs) * correct_call_pseudocount_bonus

    # sum of tilde alphas
    total_posterior_param = total_pseudocount + len(data)

    for call_true_pair in tqdm(product(rlbs, repeat=2)):
        pair_pseudocounts = pseudocounts
        if call_true_pair[0] == call_true_pair[1]:
            pair_pseudocounts += correct_call_pseudocount_bonus

        # tilde alpha
        posterior_param = pair_pseudocounts + count[call_true_pair]

        # equation (5) in write up
        log_prob = lgamma(total_posterior_param) + lgamma(1.0 + posterior_param) \
                   - lgamma(1.0 + total_posterior_param) - lgamma(posterior_param)

        distr[call_true_pair] = exp(log_prob)

    return distr


def write_trained_model(distr, file_out):
    encoding = "UTF-8"

    # header
    line_string = "called_base\tcalled_len\ttrue_base\ttrue_len\tprob\n"
    line_bytes = bytes(line_string, encoding)
    file_out.write(line_bytes)

    # write each entry in the distribution as a line
    for pair in sorted(distr.keys()):
        line_string = "\t".join(list(map(str, [pair[0][0], pair[0][1], pair[1][0], pair[1][1], distr[pair]])))
        line_bytes = bytes(line_string, encoding)
        file_out.write(line_bytes)


def make_log_conditional_memo(distr, max_run_length = 50):
    rlbs = [('-', 0)] + list(product("ACGT", range(1, max_run_length + 1)))
    memo = {}
    for called_rlb in rlbs:
        normalizing_factor = 0.0
        for true_rlb in rlbs:
            normalizing_factor += distr[(called_rlb, true_rlb)]
        for true_rlb in rlbs:
            memo[(called_rlb, true_rlb)] = log(distr[(called_rlb, true_rlb)] / normalizing_factor)
        
    return memo


def complement(base):
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


# this pileup should be a list of (base, length, strand) tuples, where strand == True
# indicates the reverse strand
def get_consensus_posterior(rlb_pileup, memo, max_run_length = 50):
    posterior = {}
    max_log_likelihood = -float_info.max
    rlbs = [('-', 0)] + list(product("ACGT", range(1, max_run_length + 1)))
    for true_rlb in rlbs:
        ll = 0.0
        for base, length, strand in rlb_pileup:
            if strand:
                called = complement(base)
                true = complement(true_rlb[0])
            else:
                called = base
                true = true_rlb[0]
            ll += memo[((called, length), (true, true_rlb[1]))]
        posterior[true_rlb] = ll
        max_log_likelihood = max(ll, max_log_likelihood)
    
    for true_rlb in rlbs:
        posterior[true_rlb] = exp(posterior[true_rlb] - max_log_likelihood)
    
    total = sum(posterior.values())    
    
    for true_rlb in rlbs:
        posterior[true_rlb] /= total
    
    return posterior


def write_joint_distribution_to_file(distribution, output_dir):
    FileManager.ensure_directory_exists(output_dir)

    datetime_string = FileManager.get_datetime_string()

    filename_prefix = "joint_distribution"
    filename = filename_prefix + "_" + datetime_string + ".tsv"
    path = os.path.join(output_dir, filename)

    with open(path, 'w') as file:
        writer = csv.writer(file, delimiter="\t")

        for pair in sorted(distribution.keys()):
            line = [pair[0][0], pair[0][1], pair[1][0], pair[1][1], distribution[pair]]

            writer.writerow(line)

    return path


def main():
    output_dir = "output/trained_jordan_model/"

    # a few by-eye tests

    # to test the file I/O
    from tempfile import TemporaryFile

    max_rle = 2

    #        called    true
    data = [(('A', 1), ('A', 2)),
            (('A', 1), ('A', 1)),
            (('A', 1), ('-', 0)),
            (('A', 1), ('A', 1)),
            (('C', 1), ('C', 2)),
            (('C', 1), ('C', 1)),
            (('C', 1), ('-', 0)),
            (('C', 1), ('C', 1)),
            (('T', 1), ('T', 2)),
            (('T', 1), ('T', 1)),
            (('T', 1), ('-', 0)),
            (('T', 1), ('T', 1)),
            (('G', 1), ('G', 2)),
            (('G', 1), ('G', 1)),
            (('G', 1), ('-', 0)),
            (('G', 1), ('G', 1)),
            (('A', 2), ('A', 2)),
            (('C', 2), ('C', 2)),
            (('G', 2), ('G', 2)),
            (('T', 2), ('T', 2)),
            (('-', 0), ('-', 0)),
            (('-', 0), ('-', 0)),
            (('A', 1), ('T', 1)),
            (('T', 1), ('A', 1)),
            (('C', 1), ('G', 1)),
            (('G', 1), ('C', 1))]

    # view and sanity check the joint distribution
    distr = train_model(data, max_rle)
    for pair in sorted(distr.keys()):
        print("{0}: {1}".format(pair, distr[pair]))

    print("should sum to 1: " + str(sum(distr.values())))

    # test the file format
    distribution_file = write_joint_distribution_to_file(distribution=distr, output_dir=output_dir)
    with open(distribution_file, "r") as file:
        for line in file:
            print(line.strip())

    # view and sanity check the conditional distributions
    memo = make_log_conditional_memo(distr, max_rle)

    for conditional_pair in sorted(memo.keys()):
        print("{0}|{1}: {2}".format(conditional_pair[1], conditional_pair[0], exp(memo[conditional_pair])))

    rlbs = [('-', 0)] + list(product("ACGT", range(1, max_rle + 1)))
    for called_rlb in rlbs:
        total = sum(exp(memo[(called_rlb, true_rlb)]) for true_rlb in rlbs)
        print(str(called_rlb) + " should sum to 1: " + str(total))

    # try the posterior inference and view results
    #          base len strand (where True -> reverse, False -> forward)
    pileup = [("A", 1, True),
              ("A", 1, False),
              ("A", 1, True),
              ("A", 1, False),
              ("A", 2, True),
              ("C", 1, False),
              ("-", 0, True)]

    posterior = get_consensus_posterior(pileup, memo, max_rle)

    for rlb in sorted(posterior.keys(), key=lambda x: posterior[x], reverse=True):
        print("{0}: {1}".format(rlb, posterior[rlb]))


if __name__ == "__main__":
    main()
