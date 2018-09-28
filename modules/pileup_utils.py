import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from handlers.FileManager import FileManager
from modules.AlignedSegmentGrabber import MAX_COVERAGE
from matplotlib import pyplot
import numpy


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


def save_run_length_training_data(output_dir, pileup_matrix, reference_matrix, pileup_repeat_matrix, reference_repeat_matrix, chromosome_name, start):
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
    numpy.savez_compressed(data_path, a=pileup_matrix, b=reference_matrix, c=pileup_repeat_matrix, d=reference_repeat_matrix)


def convert_aligned_reference_to_one_hot(reference_alignment):
    """
    given a reference sequence, generate an lx5 matrix of one-hot encodings where l=sequence length and 5 is the # of
    nucleotides, plus a null character
    :param reference_sequence:
    :return:
    """
    alignment_string = reference_alignment[0][1]

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


def plot_one_hot_tensor(tensor):
    scale = numpy.arange(0,1.0,1/5)
    scaled_tensor = tensor*scale[:,numpy.newaxis,numpy.newaxis]

    flattened_tensor = numpy.sum(scaled_tensor,axis=0)

    # print(numpy.sum(tensor, axis=0))

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