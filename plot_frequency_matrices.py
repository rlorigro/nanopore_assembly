import numpy
from matplotlib import pyplot


def load_frequency_matrix(path):
    matrix = numpy.load(path)['a']
    matrix = matrix[1:, 1:]  # trim 0 columns (for now)

    return matrix


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


def main():
    fixed_edges_path = "output/runlength_frequency_matrix/runlength_probability_matrix_2018-10-11-17-59-27.npz"
    selected_edges_path = "output/runlength_frequency_matrix/runlength_probability_matrix_2018-10-11-17-46-26.npz"

    matrix_fixed = load_frequency_matrix(fixed_edges_path)
    matrix_selected = load_frequency_matrix(selected_edges_path)

    matrix_fixed = normalize_frequency_matrix(matrix_fixed, log_scale=True)[:20,:20]
    matrix_selected = normalize_frequency_matrix(matrix_selected, log_scale=True)[:20,:20]

    fig, axes = pyplot.subplots(nrows=2)

    axes[0].imshow(matrix_selected)
    axes[1].imshow(matrix_fixed)

    axes[1].set_xlabel("Observed Runlength")
    axes[1].set_ylabel("True Runlength \n(fixed size windows)")
    axes[0].set_ylabel("True Runlength \n(anchored windows)")

    pyplot.show()
    pyplot.close()


if __name__ == "__main__":
    main()
