import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from matplotlib import pyplot
import numpy
from modules.pileup_utils import sequence_to_index, float_to_index, sequence_to_float
from scipy.misc import logsumexp

numpy.set_printoptions(precision=9, linewidth=sys.maxsize, suppress=True, threshold=sys.maxsize)

# MATRIX_PATH = "/home/ryan/code/nanopore_assembly/output/runlength_frequency_matrix/runlength_probability_matrix_2018-9-25-13-42-30.npz"   # sharp, filtered celegans
# MATRIX_PATH = "/home/ryan/code/nanopore_assembly/models/parameters/runlength_frequency_matrices_per_base_2018_10_16_12_55_47_584211.npz"     # diffuse, unfiltered celegans
# MATRIX_PATH = "/home/ryan/code/nanopore_assembly/models/parameters/runlength_frequency_matrices_per_base_2018_10_22_11_52_33_936778.npz"        # RNN filtered gaps celegans
MATRIX_PATH = "/home/ryan/code/nanopore_assembly/models/parameters/runlength_frequency_matrices_per_base_2018_11_13_9_50_5_890425.npz"  # E coli


class RunlengthClassifier:
    """
    Calculate the probability of true runlength given the following:
        1) a vector of observed runlengths for a set of aligned reads
        2) a matrix of dimensions [true_runlength, observed_runlength] containing raw true/observed frequencies
    """

    def __init__(self, path, log_scale=True):
        self.log_scale = log_scale

        self.pseudocount = 20

        self.base_frequency_matrices = self.load_base_frequency_matrices(path)    # redundant storage to troubleshoot
        self.probability_matrices = self.normalize_frequency_matrices(self.base_frequency_matrices,
                                                                      log_scale=log_scale,
                                                                      pseudocount=self.pseudocount,
                                                                      diagonal_pseudocount=self.pseudocount)
        # self.prior_vectors = self.get_prior_vector(base_frequency_matrices, log_scale=log_scale)

        self.y_maxes = [matrix.shape[0] for matrix in self.probability_matrices]
        self.x_maxes = [matrix.shape[1] for matrix in self.probability_matrices]

        self.float_to_index = float_to_index

    def plot_matrix(self, probability_matrix):
        axes = pyplot.axes()
        pyplot.imshow(probability_matrix)
        axes.set_xlabel("Observed Runlength")
        axes.set_ylabel("True Runlength")
        pyplot.show()
        pyplot.close()

    def load_base_frequency_matrices(self, path):
        matrix_labels = ["a", "g", "t", "c"]
        base_frequency_matrices = [None for base in matrix_labels]

        for base in matrix_labels:
            matrix = numpy.load(path)[base.upper()]     # toggle for stupidity
            # matrix = matrix[1:, 1:]  # trim 0 columns (for now)

            base_index = sequence_to_index[base.upper()] - 1
            base_frequency_matrices[base_index] = matrix

        # base_frequency_matrices[sequence_to_index["G"]-1] = base_frequency_matrices[sequence_to_index["T"]-1]
        # base_frequency_matrices[sequence_to_index["C"]-1] = base_frequency_matrices[sequence_to_index["T"]-1]

        return base_frequency_matrices

    def normalize_frequency_matrices(self, frequency_matrices, log_scale, pseudocount=0, diagonal_pseudocount=0):
        normalized_frequency_matrices = list()
        matrix_labels = ["a", "g", "t", "c"]

        for f,frequency_matrix in enumerate(frequency_matrices):
            frequency_matrix += pseudocount
            diagonal_mask = numpy.eye(frequency_matrix.shape[0], frequency_matrix.shape[1], dtype=numpy.bool)
            frequency_matrix[diagonal_mask] += diagonal_pseudocount

            normalized_frequencies = self.normalize_frequency_matrix(frequency_matrix=frequency_matrix,
                                                                     log_scale=log_scale)
            normalized_frequency_matrices.append(normalized_frequencies)

            self.plot_matrix(normalized_frequencies)

            # print(matrix_labels[f].upper())
            # print(normalized_frequencies)

        return normalized_frequency_matrices

    def normalize_frequency_matrix(self, frequency_matrix, log_scale):
        """
        for each true value Y, normalize observed values x such that the sum of p(x_i|Y) for all i = 1
        :param probability_matrix:
        :param log_scale:
        :return:
        """
        sum_y = numpy.sum(frequency_matrix, axis=1)

        probability_matrix = frequency_matrix / sum_y[:, numpy.newaxis]

        if log_scale:
            probability_matrix = numpy.log10(probability_matrix)

        return probability_matrix

    def log_sum_exp(self, x):
        """
        Non-log addition in log-space vector of values... doesn't work for signed addition? currently unused
        :param x:
        :return:
        """
        b = numpy.max(x[(x<sys.maxsize)])   # ignore inf values

        s = b + numpy.log(numpy.sum(numpy.exp(x-b)))

        return s

    def normalize_likelihoods(self, log_likelihood_y, max_index):
        """
        Given a vector of log likelihood values for each Y, and the index of the maximum p(y), normalize each value wrt
        the maximum: p(Y_i|x)/p(Y_max|x)
        :param log_likelihood_y:
        :param max_index:
        :return:
        """
        max_value = log_likelihood_y[max_index,:]

        normalized_likelihoods = log_likelihood_y - max_value

        return normalized_likelihoods

    def factor_repeats(self, x):
        """
        Given a vector of repeats factor them into counts for each unique repeat length
        e.g. [1,1,1,1,2,2,2] -> [1,2], [4,3]
        :param x:
        :return:
        """
        unique, inverse = numpy.unique(x, return_inverse=True)
        bincount = numpy.bincount(inverse)

        return unique, bincount

    # def apply_prior(self, y):
    #     if self.log_scale:
    #         y = y + self.prior_vector
    #     else:
    #         y = y*self.prior_vector
    #
    #     return y

    def get_base_index_from_encoding(self, float_value):
        float_value = round(float(float_value), 3)
        base_index = self.float_to_index[float_value] - 1   # no deletes allowed

        return base_index

    def predict(self, character_index, x, skip_zeros=False):
        """
        for a vector of observations x, find the product of likelihoods p(x_i|y_j) for x_i in x, for all possible Y
        values, and return the maximum value
        :param x:
        :return:
        """
        x, counts = self.factor_repeats(x)  # factor the repeats to avoid iterating probability lookups multiple times

        log_likelihood_y = numpy.zeros([self.y_maxes[character_index], 1])

        for y_j in range(0, self.y_maxes[character_index]):
            # initialize log likelihood for this (jth) y value
            log_sum = 0

            for i in range(x.shape[0]):
                # iterate x values, summing log likelihood for every p(x_i|y_j)
                x_i = x[i]
                c_i = counts[i]

                # convert to indices
                x_i = int(x_i)
                y_j = int(y_j)

                if skip_zeros:
                    if x_i == 0:
                        continue

                if x_i >= self.x_maxes[character_index]:
                    x_i = self.x_maxes[character_index] - 1

                # retrieve conditional probability for this x|y
                prob_x_i_given_y_j = self.probability_matrices[character_index][y_j, x_i]  # index NOT adjusted to account for no zeros

                # print(x_i, y_j, prob_x_i_given_y_j, 10**prob_x_i_given_y_j)

                # exponentiate by the number of independently observed repeats of this value
                log_sum += c_i*float(prob_x_i_given_y_j)

            if numpy.isnan(log_sum):    # if the frequency matrix has empty rows, consider p(y_j) to be 0
                log_sum = -numpy.inf

            # store result of log sum of likelihoods
            log_likelihood_y[y_j,0] = log_sum

        j_max = numpy.argmax(log_likelihood_y)

        normalized_posterior = self.normalize_likelihoods(log_likelihood_y=log_likelihood_y, max_index=j_max)

        # print(10**normalized_posterior)
        # print(j_max)

        return normalized_posterior, j_max

    def print_normalized_likelihoods(self, normalized_likelihoods):
        for i in range(normalized_likelihoods.shape[0]):
            print("%d:\t%.3f" % (i,float(normalized_likelihoods[i,:])))


def test():
    runlength_classifier = RunlengthClassifier(MATRIX_PATH)

    while True:
        sys.stdout.write("Enter character: \n")
        input_string = sys.stdin.readline().strip().upper()
        character_index = sequence_to_index[input_string] - 1

        sys.stdout.write("Enter space-separated repeat observations: \n")
        input_string = sys.stdin.readline()
        x = input_string.strip().split(" ")
        x = numpy.array(list(map(int, x)))

        normalized_y_log_likelihoods, y_max = runlength_classifier.predict(x=x, character_index=character_index)
        runlength_classifier.print_normalized_likelihoods(10**normalized_y_log_likelihoods)

        print("\nMost likely runlength: ", y_max)


if __name__ == "__main__":
    test()
