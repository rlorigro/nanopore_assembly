from modules.pileup_utils import sequence_to_float, sequence_to_index, trim_empty_rows, index_to_float
from modules.ConsensusCaller import ConsensusCaller
from handlers.DataLoaderRunlength import DataLoader
from handlers.FileManager import FileManager
from collections import defaultdict
from matplotlib import pyplot
import datetime
import numpy
import sys
import os

numpy.set_printoptions(linewidth=400, threshold=100000, suppress=True, precision=3)


FREQUENCY_THRESHOLD = 0.7


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


def save_numpy_matrix(output_dir, filename, matrix):
    array_file_extension = ".npz"

    # ensure chromosomal directory exists
    if not os.path.exists(output_dir):
        FileManager.ensure_directory_exists(output_dir)

    output_path_prefix = os.path.join(output_dir, filename)

    output_path = output_path_prefix + array_file_extension

    # write numpy arrays
    numpy.savez_compressed(output_path, a=matrix)


def convert_indices_to_float_encodings(indices):
    encoding = indices*0.2
    encoding = encoding.reshape([1,encoding.shape[0]])

    return encoding


def plot_runlength_distributions(runlengths):
    x_min = 0
    x_max = max(runlengths.keys()) + 1

    fig, axes = pyplot.subplots(nrows=len(runlengths), sharex=True, sharey=True)

    for k,key in enumerate(sorted(runlengths)):
        runlength_values = runlengths[key]

        step = 1
        bins = numpy.arange(x_min, x_max + step, step=step)
        frequencies, bins = numpy.histogram(runlength_values, bins=bins, normed=True)

        center = (bins[:-1] + bins[1:]) / 2 - step/2

        axes[k].bar(center, frequencies, width=step, align="center")
        axes[k].set_ylabel(str(key))
        axes[k].set_xticks(numpy.arange(x_min, x_max+1))

    axes[len(runlengths)-1].set_xlabel("Observed run length")
    pyplot.show()
    pyplot.close()


def get_runlengths(x_pileup, x_repeat, y_repeat):
    lengths = defaultdict(list)

    consensus_caller = ConsensusCaller(sequence_to_float=sequence_to_float, sequence_to_index=sequence_to_index)

    allele_frequencies = consensus_caller.get_normalized_frequencies(pileup_matrix=x_pileup)

    x_max_indices = numpy.argmax(allele_frequencies, axis=0)

    x_max_encoding = convert_indices_to_float_encodings(x_max_indices)

    x_repeat = consensus_caller.get_consensus_repeats(repeat_matrix=x_repeat, pileup_matrix=x_pileup, consensus_encoding=x_max_encoding)

    for i in range(len(y_repeat)):
        y = int(y_repeat[:,i])
        x = int(x_repeat[i])

        lengths[y].append(x)

    return lengths


def get_runlengths_from_one_hot(x_pileup, x_repeat, y_repeat):
    lengths = defaultdict(list)

    consensus_caller = ConsensusCaller(sequence_to_float=sequence_to_float, sequence_to_index=sequence_to_index)

    # consensus_repeats = consensus_caller.get_consensus_repeats_from_one_hot(repeat_matrix=x_repeat, pileup_matrix=x_pileup)
    consensus_repeats = consensus_caller.get_repeats_from_one_hot(repeat_matrix=x_repeat)

    for i in range(y_repeat.shape[-1]):
        # print(y_repeat.shape)
        y = int(y_repeat[0,0,i])

        # print(x_repeat[i])
        x = consensus_repeats[i]

        # print(x.ndim)
        if x.ndim == 1:
            lengths[y].append(x)
        else:
            # print(x)
            x = numpy.atleast_1d(x)
            # print(x)
            lengths[y].append(x)

        # if y > 8:
        #     print(y)
        #     print(x)

    return lengths


def encode_runlength_distributions_as_matrix(runlengths, log_scale=False, normalize_observed=False):
    x_min = 0
    x_max = max(runlengths.keys()) + 1
    length = x_max - x_min

    frequencies_list = list()

    for i in range(x_min,x_max):
        if i in runlengths:
            runlength_values = runlengths[i]

            step = 1
            bins = numpy.arange(x_min, x_max + step, step=step)
            frequencies, bins1 = numpy.histogram(runlength_values, bins=bins, normed=normalize_observed)
            frequencies = frequencies.reshape([1,frequencies.shape[0]])
        else:
            frequencies = numpy.zeros([1,length])

        frequencies_list.append(frequencies)

    frequencies = numpy.concatenate(frequencies_list, axis=0)

    if log_scale:
        frequencies = numpy.log10(frequencies + 1)

    return frequencies


def measure_runlengths(data_loader):
    all_runlengths = defaultdict(list)

    n_files = len(data_loader)
    n = 0

    print("testing n windows: ", n_files)

    for batch in data_loader:
        paths, x_pileup, y_pileup, x_repeat, y_repeat, reversal = batch

        # x_pileup = trim_empty_rows(x_pileup[0,:,:], background_value=sequence_to_float["-"])
        # # y_pileup = y_pileup[0,:,:]
        # x_repeat = trim_empty_rows(x_repeat[0,:,:], background_value=sequence_to_float["-"])
        # y_repeat = y_repeat[0,:,:]
        #
        # x_pileup = numpy.atleast_2d(x_pileup)
        # # y_pileup = numpy.atleast_2d(y_pileup)
        # x_repeat = numpy.atleast_2d(x_repeat)
        # y_repeat = numpy.atleast_2d(y_repeat)

        x_pileup = x_pileup[0,:,:,:]
        y_pileup = y_pileup[0,:,:,:]
        x_repeat = x_repeat[0,:,:,:]
        y_repeat = y_repeat[0,:,:,:]

        # try:
        runlengths = get_runlengths_from_one_hot(x_pileup=x_pileup, x_repeat=x_repeat, y_repeat=y_repeat)

        # except IndexError as e:
        #     print()
        #     print(e)
        #     # pyplot.imshow(x_repeat)
        #     # pyplot.show()
        #     # pyplot.close()
        #     # pyplot.imshow(x_pileup)
        #     # pyplot.show()
        #     # pyplot.close()
        #     continue

        for key in runlengths:
            runlength_values = runlengths[key]
            all_runlengths[key].extend(runlength_values)

        sys.stdout.write("\r " + str(round(n/n_files*100,3)) + "% completed")
        n += 1

        # if n > 1000:
        #     break

    sys.stdout.write("\r 100%% completed     \n")

    for key in list(all_runlengths.keys()):
        runlength_values = all_runlengths[key]
        all_runlengths[key] = numpy.concatenate(runlength_values)

    plot_runlength_distributions(all_runlengths)

    runlength_matrix = encode_runlength_distributions_as_matrix(all_runlengths, log_scale=False, normalize_observed=False)

    pyplot.imshow(normalize_frequency_matrix(runlength_matrix, log_scale=True))
    pyplot.show()

    output_dir = "output/runlength_frequency_matrix"
    datetime_string = '-'.join(list(map(str, datetime.datetime.now().timetuple()))[:-3])
    filename = "runlength_probability_matrix_" + datetime_string

    print("SAVING: ", os.path.join(output_dir, filename))

    save_numpy_matrix(output_dir=output_dir, filename=filename, matrix=runlength_matrix)


def run():
    # data_path = "/home/ryan/code/nanopore_assembly/output/celegans_chr1_1m_windows_spoa_pileup_generation_2018-9-18"    # 1 million bases in celegans chr1 scrappie
    # data_path = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_human_chr1_1mbp_2018-9-18"             # 1 million bases in human guppy
    # data_path = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_celegans_chr1_2-12Mbp_2018-9-21"       # 10 million bases in human guppy
    # data_path = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_chr5_FULL_20Mbp_2018-9-24"             # chr5 full float encoded
    # data_path = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-2-10-43-22-1-275/NC_003279.8"  # chr1 one-hot
    data_path = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_fixed_size2018-10-11-13-1-59-3-284"    # chr1 fixed size windows one-hot
    # data_path = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_anchored2018-10-11-11-20-29-3-284"    # chr1 windowed windows one-hot


    file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=data_path, file_extension=".npz")

    data_loader = DataLoader(file_paths, batch_size=1, parse_batches=False)

    lengths = measure_runlengths(data_loader)


if __name__ == "__main__":
    run()