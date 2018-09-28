import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from handlers.FileManager import FileManager
from matplotlib import pyplot
import numpy
import math


DELETE = 3
INSERT = 2
MISMATCH = 1
MATCH = 0


def plot_kernel_distribution(pdf, cdf, bins, save=False, output_dir=None, filename=None):
    n_steps = 100
    step = float(1.0/n_steps)
    center = (bins[:-1] + bins[1:]) / 2 - step / 2

    fig, axes = pyplot.subplots(nrows=2)
    axes[0].plot(cdf)
    axes[1].bar(center, pdf, width=step, align="center")
    axes[1].set_ylabel("kernel sum")

    if save:
        FileManager.ensure_directory_exists(output_dir)
        filename = filename + "_distributions.png"
        path = os.path.join(output_dir, filename)
        pyplot.savefig(path)

    else:
        pyplot.show()

    pyplot.close()


def plot_kernels_and_column_frequencies(kernel_sums, passing_indices, column_frequencies, slice_range=None, save=False, output_dir=None, filename=None):
    if slice_range is not None:
        kernel_sums = kernel_sums[:,slice_range[0]:slice_range[1]]
        passing_indices = passing_indices[:,slice_range[0]:slice_range[1]]
        column_frequencies = column_frequencies[:,slice_range[0]:slice_range[1]]

        kernel_sums.reshape(1, slice_range[1] - slice_range[0])
        passing_indices.reshape(1, slice_range[1] - slice_range[0])
        column_frequencies.reshape(column_frequencies.shape[0], slice_range[1] - slice_range[0])

    fig, axes = pyplot.subplots(nrows=3, sharex=True)
    fig.set_size_inches(16,4)
    axes[0].imshow(passing_indices)
    axes[1].imshow(kernel_sums)
    axes[2].imshow(column_frequencies)

    axes[0].set_ylabel("Thresholded")
    axes[1].set_ylabel("Convolution")
    axes[2].set_ylabel("Frequencies")

    axes[0].set_yticklabels([])
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])

    axes[0].set_yticks([])
    axes[1].set_yticks([])
    axes[2].set_yticks([])

    if save:
        FileManager.ensure_directory_exists(output_dir)
        filename = filename + "_kernels.png"
        path = os.path.join(output_dir, filename)
        pyplot.savefig(path)

    else:
        pyplot.show()

    pyplot.close()


def select_window_edges(passing_indices, reference_start_position):
    length = passing_indices.shape[1]

    windows = list()

    # find first passing index
    block_start = 0
    block_end = 0
    anchor_position = None
    previously_passing = False

    for i in range(length):
        currently_passing = bool(passing_indices[0,i])

        if currently_passing and not previously_passing:
            block_start = i

        if not currently_passing and previously_passing:
            block_end = i - 1

            if anchor_position is not None:
                window_start = reference_start_position + anchor_position + 1
                current_anchor_position = math.floor((block_start + block_end)/2)

                if current_anchor_position - anchor_position > 20:
                    anchor_position = current_anchor_position
                    window_end = reference_start_position + anchor_position
                    windows.append([window_start, window_end])

            else:
                anchor_position = math.floor((block_start + block_end)/2)

        previously_passing = currently_passing

    return windows


def get_threshold(cdf, p, n_steps):
    index = numpy.argmax((cdf>p))
    threshold = index/n_steps
    return threshold


def approximate_pdf_and_cdf(array, n_steps=100):
    step = float(1.0 / n_steps)
    bins = numpy.arange(0, 1 + step, step=step)
    pdf, bins = numpy.histogram(array, bins=bins, normed=False)

    cdf = numpy.cumsum(pdf)
    cdf = cdf/cdf[-1]

    return pdf, cdf, bins


def get_kernel_sums(matrix, normalize=True):
    kernel = numpy.array([0.25,0.5,1,1,0.5,0.25])
    kernel_sums = numpy.convolve(matrix, kernel, mode="same")
    kernel_sums = kernel_sums.reshape([1,kernel_sums.shape[0]])

    if normalize:
        c = numpy.sum(kernel)
        kernel_sums = kernel_sums/c

    return kernel_sums


def get_column_frequencies(sam_file, reference_sequence, chromosome_name, test_window):
    column_frequencies = list()

    for pileup_column in sam_file.pileup(chromosome_name, test_window[0], test_window[1]):
        if test_window[0] < pileup_column.pos < test_window[1]:
            ref_position = pileup_column.reference_pos
            ref_character = reference_sequence[ref_position]
            coverage = pileup_column.nsegments

            frequencies = numpy.zeros([4,1])

            n = 0
            for pileup_read in pileup_column.pileups:
                # delete
                if pileup_read.is_del:
                    frequencies[DELETE,0] += 1

                # insert
                elif pileup_read.indel > 0:
                    frequencies[INSERT,0] += 1

                else:
                    read_character = pileup_read.alignment.query_sequence[pileup_read.query_position]

                    if read_character == ref_character:
                        # match
                        frequencies[MATCH,0] += 1
                    else:
                        # mismatch
                        frequencies[MISMATCH,0] += 1

                n += 1

            frequencies = frequencies/coverage
            sum_of_metrics = numpy.sum(frequencies)

            if round(float(sum_of_metrics),6) != 1.0:
                print("WARNING: sum of column characters != 1.0:\t", sum_of_metrics)

            # print('\t'.join(map(str,[coverage, sum_of_metrics, n_match, n_deletes, n_inserts, n_mismatch])))

            column_frequencies.append(frequencies)

    return column_frequencies
