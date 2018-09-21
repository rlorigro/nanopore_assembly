from matplotlib import pyplot
import numpy
import sys
from scipy.misc import logsumexp

numpy.set_printoptions(precision=2, linewidth=400, suppress=True)


class RunlengthClassifier():
    """
    Given a matrix of dimensions [true_runlength, observed_runlength] calculate the probability of true runlength
    given a vector of observed runlengths
    """

    def __init__(self, frequency_matrix, log_scale=True):
        self.log_scale = log_scale

        self.frequency_matrix = frequency_matrix    # redundant storage for troubleshooting
        self.probability_matrix = self.normalize_frequency_matrix(self.frequency_matrix, log_scale=True)

        self.y_max = self.probability_matrix.shape[0]
        self.x_max = self.probability_matrix.shape[1]

    def normalize_frequency_matrix(self, probability_matrix, log_scale):
        """
        for each true value Y, normalize observed values x such that the sum of p(x_i|Y) for all i = 1
        :param probability_matrix:
        :param log_scale:
        :return:
        """
        sum_y = numpy.sum(probability_matrix, axis=1)

        probability_matrix = probability_matrix / sum_y[:, numpy.newaxis]

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

    def predict(self, x):
        """
        for a vector of observations x, find the product of likelihoods p(x_i|y_j) for x_i in x, for all possible Y
        values, and return the maximum value
        :param x:
        :return:
        """

        log_likelihood_y = numpy.zeros([self.y_max, 1])

        for y_j in range(0, self.y_max):
            # initialize log likelihood for this (jth) y value
            log_sum = 0

            for i in range(x.shape[0]):
                # iterate x values, summing log likelihood for every p(x_i|y_j)
                x_i = x[i]

                prob_x_i_given_y_j = self.probability_matrix[y_j - 1, x_i - 1]  # index adjusted to account for no zeros

                log_sum += float(prob_x_i_given_y_j)

                # print(x_i, y_j, prob_x_i_given_y_j)

            log_likelihood_y[y_j,0] = log_sum

        j_max = numpy.argmax(log_likelihood_y)

        normalized_y_log_likelihoods = self.normalize_likelihoods(log_likelihood_y=log_likelihood_y, max_index=j_max)

        return normalized_y_log_likelihoods, j_max

    def print_normalized_likelihoods(self, normalized_likelihoods):
        for i in range(normalized_likelihoods.shape[0]):
            print("%d:\t%.3f" % (i,float(normalized_likelihoods[i,:])))

def test():
    matrix_path = "models/parameters/runlength_frequency_matrix_2018-9-20-14-21-6.npz"
    matrix = numpy.load(matrix_path)['a']
    matrix = matrix[1:,1:]   # trim 0 columns (for now)

    print(matrix)

    runlength_classifier = RunlengthClassifier(matrix)

    while True:
        sys.stdout.write("Enter space-separated repeat observations: \n")

        input_string = sys.stdin.readline()
        x = input_string.strip().split(" ")
        x = numpy.array(list(map(int, x)))

        normalized_y_log_likelihoods, y_max = runlength_classifier.predict(x)

        runlength_classifier.print_normalized_likelihoods(10**normalized_y_log_likelihoods)
        print(y_max)


if __name__ == "__main__":
    test()