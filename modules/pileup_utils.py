import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from handlers.FileManager import FileManager
from modules.AlignedSegmentGrabber import MAX_COVERAGE
from matplotlib import pyplot
import torch
import numpy

scale = numpy.arange(0, 1.0, 1 / 5)

sequence_to_float = {"-":0.02,
                     "A":0.2,
                     "G":0.4,
                     "T":0.6,
                     "C":0.8}

sequence_to_index = {"-":0,
                     "A":1,
                     "G":2,
                     "T":3,
                     "C":4}

A,G,T,C = 1,2,3,4

index_to_float = [0.02, 0.2, 0.4, 0.6, 0.8]

index_to_sequence = ["-", "A", "G", "T", "C"]

float_to_index = {0.02:0, 0.2:1, 0.4:2, 0.6:3, 0.8:4}


def complement(base):
    if base == "A":
        complement = "T"
    elif base == "T":
        complement = "A"
    elif base == "G":
        complement = "C"
    elif base == "C":
        complement = "G"
    else:
        complement = "-"

    return complement


def label_pileup_encoding_plot(matrix, axis):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = numpy.round(matrix[i,j],3)

            if value == 0:
                index = 0
            else:
                index = float_to_index[value]

            character = index_to_sequence[index]

            axis.text(j,i,character, ha="center", va="center", fontsize=6)


def label_repeat_encoding_plot(matrix, axis):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i,j]

            axis.text(j,i,str(int(value)), ha="center", va="center", fontsize=6)


def plot_runlength_prediction_stranded(x_pileup, x_repeat, y_pileup, y_repeat, reversal, save_path=None, title="", squeeze=False, label=True, show_reversal=True):
    if type(x_pileup) == torch.Tensor:
        x_pileup = x_pileup.data.numpy()

    if type(x_repeat) == torch.Tensor:
        x_repeat = x_repeat.data.numpy()

    if type(y_pileup) == torch.Tensor:
        y_pileup = y_pileup.data.numpy()

    if type(y_repeat) == torch.Tensor:
        y_repeat = y_repeat.data.numpy()

    if type(reversal) == torch.Tensor:
        reversal = reversal.data.numpy()

    print()
    print(x_pileup.shape)
    print(y_pileup.shape)
    print(x_repeat.shape)
    print(y_repeat.shape)
    print(reversal.shape)

    x_pileup = x_pileup[:, :]
    x_repeat = x_repeat[:, :]
    reversal = reversal[:, :]

    if squeeze:
        x_pileup.squeeze()
        x_repeat.squeeze()
        reversal.squeeze()

    x_pileup_ratio = x_pileup.shape[-2]/y_pileup.shape[-2]
    y_pileup_ratio = 1
    x_repeat_ratio = x_repeat.shape[-2]/y_pileup.shape[-2]
    y_repeat_ratio = y_repeat.shape[-2]/y_pileup.shape[-2]
    reversal_ratio = reversal.shape[-2]/y_pileup.shape[-2]

    n_rows = 4
    height_ratios = [y_pileup_ratio, x_pileup_ratio, y_repeat_ratio, x_repeat_ratio]

    if show_reversal:
        n_rows = 5
        height_ratios.append(reversal_ratio)

    fig, axes = pyplot.subplots(nrows=n_rows, gridspec_kw={'height_ratios': height_ratios})

    if label:
        label_pileup_encoding_plot(matrix=y_pileup, axis=axes[0])
        label_pileup_encoding_plot(matrix=x_pileup, axis=axes[1])
        label_repeat_encoding_plot(matrix=y_repeat, axis=axes[2])
        label_repeat_encoding_plot(matrix=x_repeat, axis=axes[3])

    axes[0].imshow(y_pileup)
    axes[1].imshow(x_pileup)
    axes[2].imshow(y_repeat)
    axes[3].imshow(x_repeat)

    axes[0].set_ylabel("y")
    axes[1].set_ylabel("x sequence")
    axes[2].set_ylabel("y")
    axes[3].set_ylabel("x repeats")

    if show_reversal:
        axes[4].imshow(reversal)
        axes[4].set_ylabel("reversal")

    axes[0].set_title(title)

    if save_path is not None:
        pyplot.savefig(save_path)
    else:
        pyplot.show()
        pyplot.close()


def plot_runlength_prediction(x_pileup, x_repeat, y_pileup, y_repeat, save_path=None, title="", squeeze=False, label=True):
    if type(x_pileup) == torch.Tensor:
        x_pileup = x_pileup.data.numpy()

    if type(x_repeat) == torch.Tensor:
        x_repeat = x_repeat.data.numpy()

    if type(y_pileup) == torch.Tensor:
        y_pileup = y_pileup.data.numpy()

    if type(y_repeat) == torch.Tensor:
        y_repeat = y_repeat.data.numpy()

    x_pileup = x_pileup[:, :]
    x_repeat = x_repeat[:, :]

    if squeeze:
        x_pileup.squeeze()
        x_repeat.squeeze()

    print()
    print(x_pileup.shape)
    print(y_pileup.shape)
    print(x_repeat.shape)
    print(y_repeat.shape)

    x_pileup_ratio = x_pileup.shape[-2]/y_pileup.shape[-2]
    y_pileup_ratio = 1
    x_repeat_ratio = x_repeat.shape[-2]/y_pileup.shape[-2]
    y_repeat_ratio = y_repeat.shape[-2]/y_pileup.shape[-2]

    fig, axes = pyplot.subplots(nrows=4, gridspec_kw={'height_ratios': [y_pileup_ratio,
                                                                        x_pileup_ratio,
                                                                        y_repeat_ratio,
                                                                        x_repeat_ratio]})

    if label:
        label_pileup_encoding_plot(matrix=y_pileup, axis=axes[0])
        label_pileup_encoding_plot(matrix=x_pileup, axis=axes[1])
        label_repeat_encoding_plot(matrix=y_repeat, axis=axes[2])
        label_repeat_encoding_plot(matrix=x_repeat, axis=axes[3])

    axes[0].imshow(y_pileup)
    axes[1].imshow(x_pileup)
    axes[2].imshow(y_repeat)
    axes[3].imshow(x_repeat)

    axes[0].set_ylabel("y")
    axes[1].set_ylabel("x sequence")
    axes[2].set_ylabel("y")
    axes[3].set_ylabel("x repeats")

    axes[0].set_title(title)

    if save_path is not None:
        pyplot.savefig(save_path)
    else:
        pyplot.show()
        pyplot.close()


def get_joint_base_runlength_observations_vs_truth(x_pileup, x_repeat, y_pileup, y_repeat, reversal, columnar=False, path=None):
    # input shape: (n_channels, height, width)
    n_channels, height, width = x_pileup.shape

    if columnar:
        observations_vs_truths = [list() for i in range(width)]
    else:
        observations_vs_truths = list()

    DEBUG_WEIRD_IMAGE = False

    for w in range(width):
        diff = 0

        # print("\ncolumn: ", w)
        for h in range(height):
            # reference has only one "coverage"
            true_base_index = int(numpy.argmax(y_pileup[:, 0, w]))
            true_base = index_to_sequence[true_base_index]
            true_repeat = int(y_repeat[0, w])

            observed_base_index = int(numpy.argmax(x_pileup[:,h,w]))
            observed_base = index_to_sequence[observed_base_index]
            observed_repeat = int(x_repeat[0,h,w])
            base_reversed = bool(reversal[0,h,w])

            diff += abs(true_repeat-observed_repeat)

            if base_reversed:
                observed_base = complement(observed_base)
                true_base = complement(true_base)

            observation_vs_truth = ((observed_base, observed_repeat),(true_base, true_repeat))
            if columnar:
                observations_vs_truths[w].append(observation_vs_truth)
            else:
                observations_vs_truths.append(observation_vs_truth)


            # if observed_base == "C" and observed_repeat == 4 and true_repeat == 16:
            #     DEBUG_WEIRD_IMAGE = True

    # if DEBUG_WEIRD_IMAGE:
    #     print(path)
    #     x_pileup_flat = flatten_one_hot_tensor(x_pileup)
    #     y_pileup_flat = flatten_one_hot_tensor(y_pileup)
    #     plot_runlength_prediction_stranded(x_pileup=x_pileup_flat,
    #                                        x_repeat=x_repeat.squeeze(),
    #                                        y_pileup=y_pileup_flat,
    #                                        y_repeat=y_repeat,
    #                                        reversal=reversal.squeeze(),
    #                                        show_reversal=False,
    #                                        label=True)

    # if width > 500:
    #     x_pileup_flat = flatten_one_hot_tensor(x_pileup)
    #     y_pileup_flat = flatten_one_hot_tensor(y_pileup)
    #     plot_runlength_prediction_stranded(x_pileup=x_pileup_flat,
    #                                        x_repeat=x_repeat.squeeze(),
    #                                        y_pileup=y_pileup_flat,
    #                                        y_repeat=y_repeat,
    #                                        reversal=reversal.squeeze(),
    #                                        label=False)

    return observations_vs_truths


def get_joint_base_runlength_observations(x_pileup, x_repeat, reversal, columnar=False, use_complement=False, max_runlength=sys.maxsize):
    # input shape: (n_channels, height, width)
    n_channels, height, width = x_pileup.shape

    if reversal is None:
        reversal = numpy.zeros([1, height, width], dtype=numpy.bool)

    if columnar:
        observations = [list() for i in range(width)]
    else:
        observations = list()

    for w in range(width):
        for h in range(height):
            # reference has only one "coverage"
            observed_base_index = int(numpy.argmax(x_pileup[:, h, w]))
            observed_base = index_to_sequence[observed_base_index]
            observed_repeat = int(x_repeat[0, h, w])
            base_reversed = bool(reversal[0, h, w])

            # print(observed_base_index, observed_base, observed_repeat, true_base_index, true_base, true_repeat, base_reversed)
            # if use_complement:
            #     if base_reversed:
            #         observed_base = complement(observed_base)

            if observed_repeat > max_runlength:
                observed_repeat = max_runlength

            observation = (observed_base, observed_repeat, base_reversed)

            if columnar:
                observations[w].append(observation)
            else:
                observations.append(observation)

    return observations


def visualize_matrix(matrix):
    pyplot.imshow(matrix, cmap="viridis")
    pyplot.show()


def plot_encodings(pileup_matrix, reference_matrix):
    ratio = pileup_matrix.shape[0] / reference_matrix.shape[0]
    fig, axes = pyplot.subplots(nrows=2, gridspec_kw={'height_ratios': [1, ratio]})

    x_data = pileup_matrix.squeeze()
    y_target_data = reference_matrix.squeeze()

    axes[1].imshow(x_data)
    axes[0].imshow(y_target_data)

    axes[1].set_ylabel("x")
    axes[0].set_ylabel("y")

    pyplot.show()
    pyplot.close()


def plot_collapsed_encodings(pileup_matrix, pileup_repeat_matrix, reference_matrix, reference_repeat_matrix):
    ratio = pileup_matrix.shape[0] / reference_matrix.shape[0]
    fig, axes = pyplot.subplots(nrows=4, gridspec_kw={'height_ratios': [1, ratio, 1, ratio]})

    pileup_matrix = pileup_matrix.squeeze()
    pileup_repeat_matrix = pileup_repeat_matrix.squeeze()
    # reference_matrix = reference_matrix.squeeze()
    # reference_repeat_matrix = reference_repeat_matrix.squeeze()

    axes[3].imshow(pileup_repeat_matrix)
    axes[2].imshow(reference_repeat_matrix)
    axes[1].imshow(pileup_matrix)
    axes[0].imshow(reference_matrix)

    axes[3].set_ylabel("repeats")
    axes[2].set_ylabel("ref repeats")
    axes[1].set_ylabel("nucleotide")
    axes[0].set_ylabel("reference")

    pyplot.show()
    pyplot.close()


def trim_empty_rows(matrix, background_value):
    h, w = matrix.shape

    sums = numpy.sum(matrix, axis=1).round(3)
    mask = (sums > round(background_value * w, 3))

    matrix = matrix[mask, :].squeeze()

    return matrix


def save_training_data(output_dir, pileup_matrix, reference_matrix, chromosome_name, start):
    array_file_extension = ".npz"

    # ensure chromosomal directory exists
    chromosomal_output_dir = os.path.join(output_dir, chromosome_name)
    if not os.path.exists(chromosomal_output_dir):
        FileManager.ensure_directory_exists(chromosomal_output_dir)

    # generate unique filename and path
    filename = chromosome_name + "_" + str(start)

    output_path_prefix = os.path.join(chromosomal_output_dir, filename)

    data_path = output_path_prefix + "_matrix" + array_file_extension

    # write numpy arrays
    numpy.savez_compressed(data_path, a=pileup_matrix, b=reference_matrix)


def save_run_length_training_data(output_dir, pileup_matrix, reference_matrix, pileup_repeat_matrix, reference_repeat_matrix, reversal_matrix, chromosome_name, start):
    array_file_extension = ".npz"

    # ensure chromosomal directory exists
    chromosomal_output_dir = os.path.join(output_dir, chromosome_name)
    if not os.path.exists(chromosomal_output_dir):
        FileManager.ensure_directory_exists(chromosomal_output_dir)

    # generate unique filename and path
    filename = chromosome_name + "_" + str(start)

    output_path_prefix = os.path.join(chromosomal_output_dir, filename)

    data_path = output_path_prefix + "_matrix" + array_file_extension

    # write numpy arrays
    numpy.savez_compressed(data_path,
                           x_pileup=pileup_matrix,
                           y_pileup=reference_matrix,
                           x_repeat=pileup_repeat_matrix,
                           y_repeat=reference_repeat_matrix,
                           reversal=reversal_matrix)


def convert_aligned_reference_to_one_hot(alignment_string):
    """
    given a reference sequence, generate an lx5 matrix of one-hot encodings where l=sequence length and 5 is the # of
    nucleotides, plus a null character
    :param reference_sequence:
    :return:
    """

    length = len(alignment_string)
    matrix = numpy.zeros([5,length])

    for c,character in enumerate(alignment_string):
        index = sequence_to_index[character]

        matrix[index,c] = 1

    return matrix


def convert_alignments_to_matrix(alignments, fixed_coverage=True):
    """
    For a list of alignment strings, generate a matrix of encoded bases in float format from 0-1
    :param alignments:
    :return:
    """

    if fixed_coverage:
        n = MAX_COVERAGE
    else:
        n = len(alignments)

    m = len(alignments[0][1])

    matrix = numpy.zeros([n, m]) + sequence_to_float["-"]

    for a,alignment in enumerate(alignments):
        read_id, alignment_string = alignment

        for b,character in enumerate(alignment_string):
            matrix[a,b] = sequence_to_float[character]

    matrix = numpy.atleast_2d(matrix)

    return matrix


def convert_collapsed_alignments_to_one_hot_tensor(alignments, repeats, fixed_coverage=True, numpy_type=numpy.float64):
    if fixed_coverage:
        n = MAX_COVERAGE
    else:
        n = len(alignments)
    m = len(alignments[0])
    c = len(index_to_sequence)

    base_matrix = numpy.zeros([c,n,m], dtype=numpy_type)
    repeat_matrix = numpy.zeros([1,n,m])

    for n_index,alignment in enumerate(alignments):
        repeat_index = 0

        for m_index,character in enumerate(alignment):
            c_index = sequence_to_index[character]
            base_matrix[c_index,n_index,m_index] = 1

            if character != "-":
                repeat_matrix[0,n_index,m_index] = repeats[n_index][repeat_index]
                repeat_index += 1

    if base_matrix.ndim < 3:
        base_matrix = base_matrix.reshape([c,n,m])

    if repeat_matrix.ndim < 3:
        repeat_matrix = repeat_matrix.reshape([1,n,m])

    return base_matrix, repeat_matrix


def convert_collapsed_alignments_to_one_hot_tensor(alignments, repeats, fixed_coverage=True, numpy_type=numpy.float64):
    if fixed_coverage:
        n = MAX_COVERAGE
    else:
        n = len(alignments)
    m = len(alignments[0])
    c = len(index_to_sequence)

    base_matrix = numpy.zeros([c,n,m], dtype=numpy_type)
    repeat_matrix = numpy.zeros([1,n,m])

    for n_index,alignment in enumerate(alignments):
        repeat_index = 0

        for m_index,character in enumerate(alignment):
            c_index = sequence_to_index[character]
            base_matrix[c_index,n_index,m_index] = 1

            if character != "-":
                repeat_matrix[0,n_index,m_index] = repeats[n_index][repeat_index]
                repeat_index += 1

    if base_matrix.ndim < 3:
        base_matrix = base_matrix.reshape([c,n,m])

    if repeat_matrix.ndim < 3:
        repeat_matrix = repeat_matrix.reshape([1,n,m])

    return base_matrix, repeat_matrix


def convert_collapsed_alignments_to_matrix(alignments, character_counts, fixed_coverage=True, numpy_type=numpy.float64):
    """
    For a list of alignment strings, generate a matrix of encoded bases in float format from 0-1
    :param alignments:
    :return:
    """
    if fixed_coverage:
        n = MAX_COVERAGE
    else:
        n = len(alignments)
    m = len(alignments[0])

    base_matrix = numpy.zeros([n, m], dtype=numpy_type) + sequence_to_float["-"]
    repeat_matrix = numpy.zeros([n,m])

    for a,alignment in enumerate(alignments):
        c = 0

        for b,character in enumerate(alignment):
            base_matrix[a,b] = sequence_to_float[character]

            if character != "-":
                repeat_matrix[a,b] = character_counts[a][c]
                c += 1

    base_matrix = numpy.atleast_2d(base_matrix)
    repeat_matrix = numpy.atleast_2d(repeat_matrix)

    return base_matrix, repeat_matrix


def convert_reversal_statuses_to_integer_matrix(reverse_statuses, pileup_matrix):
    n_channels, height, width = pileup_matrix.shape

    reversal_matrix = numpy.zeros([height, width])
    non_blank_mask = (pileup_matrix[0, :, :] == 0)

    for r,reversal_status in enumerate(reverse_statuses):
        reversal_matrix[r,:] = reversal_status

    reversal_matrix = numpy.logical_and(non_blank_mask, reversal_matrix)

    # fig, axes = pyplot.subplots(nrows=2)
    # axes[0].imshow(reversal_matrix)
    # axes[1].imshow(flatten_one_hot_tensor(pileup_matrix))
    # pyplot.show()
    # pyplot.close()

    return reversal_matrix


def flatten_one_hot_tensor(tensor):
    # tensor shape:
    scaled_tensor = tensor*scale[:,numpy.newaxis,numpy.newaxis]

    flattened_tensor = numpy.sum(scaled_tensor,axis=0)

    return flattened_tensor


def plot_one_hot_tensor(tensor):

    flattened_tensor = flatten_one_hot_tensor(tensor)

    pyplot.imshow(flattened_tensor)
    pyplot.show()
    pyplot.close()


def test():
    test_sequences = ["AGGTTTCCCC",
                      "GGTTTCCCC",
                      "ATTTCCCC",
                      "AGGCCCC",
                      "AGGTTT"]

    sequences, repeats = collapse_repeats(test_sequences)
    alignments = get_spoa_alignment_no_ref(sequences)

    print(alignments)
    print(sequences)
    print(repeats)

    base_matrix, repeat_matrix = convert_collapsed_alignments_to_matrix(alignments=alignments,
                                                                        character_counts=repeats,
                                                                        fixed_coverage=False)

    print(base_matrix)
    print(repeat_matrix)

    test_alignments = ["AGTC",
                       "-GTC",
                       "A-TC",
                       "AG-C",
                       "AGT-"]

    test_repeats = [[1,2,3,4],
                    [2,3,4],
                    [1,3,4],
                    [1,2,4],
                    [1,2,3]]

    base_tensor, repeat_tensor = convert_collapsed_alignments_to_one_hot_tensor(alignments=test_alignments,
                                                                                repeats=test_repeats,
                                                                                fixed_coverage=False)

    print(base_tensor.shape, repeat_tensor.shape)
    print(repeat_tensor.squeeze())

    plot_one_hot_tensor(base_tensor)

    pyplot.imshow(repeat_tensor.squeeze())
    pyplot.show()
    pyplot.close()


if __name__ == "__main__":
    from modules.alignment_utils import get_spoa_alignment_no_ref, collapse_repeats
    test()