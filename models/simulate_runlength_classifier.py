import numpy
import sys
import math
from matplotlib import pyplot
numpy.set_printoptions(precision=2, linewidth=400, suppress=True)


class RunlengthClassifier:
    """
    Given a matrix of dimensions [true_runlength, observed_runlength] calculate the probability of true runlength
    given a vector of observed runlengths
    """

    def __init__(self, frequency_matrix, log_scale=True):
        self.log_scale = log_scale

        self.frequency_matrix = frequency_matrix    # redundant storage for troubleshooting
        self.probability_matrix = self.normalize_frequency_matrix(self.frequency_matrix, log_scale=log_scale)

        self.y_max = self.probability_matrix.shape[0]
        self.x_max = self.probability_matrix.shape[1]

    def plot_matrix(self, probability_matrix):
        axes = pyplot.axes()
        pyplot.imshow(probability_matrix)
        axes.set_xlabel("Observed Runlength")
        axes.set_ylabel("True Runlength")
        pyplot.show()
        pyplot.close()

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

    def predict(self, x, counts):
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
                c_i = counts[i]

                # convert to indices
                x_i = int(x_i)
                y_j = int(y_j)

                if x_i > self.x_max:
                    x_i = self.x_max

                # retrieve conditional probability for this x and y
                prob_x_i_given_y_j = self.probability_matrix[y_j - 1, x_i - 1]  # index adjusted to account for no zeros

                #
                log_sum += c_i*float(prob_x_i_given_y_j)

            if numpy.isnan(log_sum):    # if the frequency matrix has empty rows, consider p(y_j) to be 0
                log_sum = -numpy.inf

            # store result of log sum of likelihoods
            log_likelihood_y[y_j,0] = log_sum

        j_max = numpy.argmax(log_likelihood_y)

        normalized_y_log_likelihoods = self.normalize_likelihoods(log_likelihood_y=log_likelihood_y, max_index=j_max)

        # print(j_max)

        return normalized_y_log_likelihoods, j_max

    def print_normalized_likelihoods(self, normalized_likelihoods):
        for i in range(normalized_likelihoods.shape[0]):
            print("%d:\t%.3f" % (i,float(normalized_likelihoods[i,:])))


def parse_input_as_factored_repeats(input_string):
    input = input_string.strip().split(" ")
    input = numpy.array(list(map(int, input)))

    length = len(input)

    if length % 2 != 0:
        exit("ERROR: input incorrect length, likely not formatted as factored repeats")

    length = int(length/2)

    x = numpy.zeros([length])
    counts = numpy.zeros([length])

    x_set = set()

    for i,x_i in enumerate(input):
        x_i = int(x_i)
        offset = i % 2
        i_destination = math.floor(i/2)

        if offset == 0:
            if x_i not in x_set:
                x[i_destination] = x_i
                x_set.add(x_i)
            else:
                exit("ERROR: duplicate repeat value detected: %d" % x_i)

        else:
            counts[i_destination] = x_i

    return x, counts


def test():
    matrix = numpy.array([[10179639, 137582, 19848, 3983, 789, 586, 343, 163, 98, 38, 18, 13, 9, 5, 5, 1, 1, 0, 2, 0, 3, ],
                          [113016, 1999397, 29116, 3979, 842, 278, 125, 62, 50, 15, 19, 10, 4, 4, 1, 1, 1, 0, 0, 1, 1, ],
                          [21259, 39549, 780217, 12777, 1861, 483, 158, 51, 27, 23, 5, 4, 3, 1, 1, 1, 0, 0, 0, 0, 0, ],
                          [6080, 7520, 19237, 319121, 6022, 1357, 330, 108, 55, 10, 4, 5, 1, 2, 1, 0, 0, 2, 0, 1, 1, ],
                          [1762, 1685, 3000, 8022, 91774, 2941, 635, 202, 62, 23, 10, 5, 1, 2, 0, 0, 0, 1, 0, 0, 1, ],
                          [782, 455, 703, 1791, 2996, 37700, 1365, 369, 89, 28, 7, 6, 3, 2, 1, 1, 0, 0, 0, 0, 1, ],
                          [346, 201, 231, 364, 699, 1094, 15635, 537, 163, 53, 21, 6, 5, 4, 1, 0, 1, 1, 0, 0, 1, ],
                          [150, 88, 83, 98, 192, 339, 531, 7620, 284, 67, 29, 27, 9, 2, 1, 1, 0, 0, 0, 0, 0, ],
                          [88, 35, 27, 43, 46, 91, 157, 233, 3058, 124, 37, 15, 2, 1, 0, 0, 0, 0, 0, 0, 0, ],
                          [38, 18, 14, 16, 12, 25, 36, 64, 85, 1245, 45, 15, 4, 3, 1, 0, 1, 1, 0, 0, 1, ],
                          [17, 6, 5, 7, 6, 6, 10, 20, 21, 42, 425, 25, 5, 1, 0, 0, 0, 1, 0, 0, 0, ],
                          [4, 3, 0, 1, 2, 3, 2, 2, 9, 6, 8, 191, 4, 4, 0, 0, 0, 0, 0, 0, 0, ],
                          [4, 4, 1, 1, 1, 0, 0, 1, 3, 4, 7, 11, 131, 4, 1, 0, 0, 1, 0, 0, 0, ],
                          [1, 0, 0, 1, 1, 0, 0, 1, 2, 1, 1, 1, 1, 70, 1, 0, 2, 0, 2, 0, 0, ],
                          [1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 30, 0, 0, 0, 0, 0, 0, ],
                          [2, 1, 1, 0, 0, 1, 0, 0, 0, 2, 0, 4, 2, 3, 4, 83, 1, 1, 0, 0, 0, ],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 7, 0, 0, 0, 0, ],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, ],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, ]])

    print(matrix)

    runlength_classifier = RunlengthClassifier(matrix)

    while True:
        sys.stdout.write("Enter space-separated repeat observations: \n")

        input_string = sys.stdin.readline()

        x, counts = parse_input_as_factored_repeats(input_string)

        normalized_y_log_likelihoods, y_max = runlength_classifier.predict(x, counts)

        runlength_classifier.print_normalized_likelihoods(10**normalized_y_log_likelihoods)

        print("\nMost likely runlength: ", y_max)


if __name__ == "__main__":
    test()