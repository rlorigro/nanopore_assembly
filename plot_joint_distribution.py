from models.JointClassifier import JOINT_DISTRIBUTION_PATH
from matplotlib import pyplot
import pickle
import numpy


def normalize_frequency_matrix(frequency_matrix, log_scale):
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


def load_pickled_distribution(path):
    with open(path, 'rb') as pickle_file:
        distribution = pickle.load(pickle_file)

    return distribution


def main():
    distribution = load_pickled_distribution("/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/distribution/distribution_2018_10_9_19_12_45_654730.pkl")

    # print(distribution.keys())
    max_runlength = 50
    for base in ["A", "G", "T", "C"]:
        base_self_distribution = numpy.zeros([max_runlength, max_runlength])

        for r_x, observed_repeat in enumerate(range(1, max_runlength+1)):
            for r_y, true_repeat in enumerate(range(1, max_runlength+1)):
                key = ((base, observed_repeat),(base, true_repeat))

                probability = distribution[key]

                base_self_distribution[r_y,r_x] = probability

        base_self_distribution = normalize_frequency_matrix(base_self_distribution, log_scale=True)
        pyplot.title(base+":"+ base +" Log probabilities")
        pyplot.imshow(base_self_distribution)
        pyplot.show()
        pyplot.close()


if __name__ == "__main__":
    main()

