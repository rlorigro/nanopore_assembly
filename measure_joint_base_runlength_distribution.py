from models.JointClassifierTrainer import train_model, JointClassifierTrainer
from modules.pileup_utils import sequence_to_float, sequence_to_index, index_to_sequence, get_joint_base_runlength_observations_vs_truth
from modules.GapFilterer import GapFilterer
from modules.ConsensusCaller import ConsensusCaller
from handlers.DataLoaderRunlength import DataLoader
from handlers.FileManager import FileManager
from collections import defaultdict
from matplotlib import pyplot
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
import random
import pickle
import numpy
import sys
import os
import gc

numpy.set_printoptions(linewidth=400, threshold=100000, suppress=True, precision=3)

FREQUENCY_THRESHOLD = 0.7


def plot_frequency_matrices(frequency_matrices, cutoff=30):
    for base in ["A", "G", "T", "C"]:
        matrix = frequency_matrices[base]
        print(numpy.sum(matrix))

        matrix = matrix[:cutoff,:cutoff]
        normalized_matrix = normalize_frequency_matrix(frequency_matrix=matrix, log_scale=True)

        axes = pyplot.axes()

        for n in range(matrix.shape[0]):
            for m in range(matrix.shape[1]):
                pyplot.text(m,n,str(int(matrix[n,m])), fontsize=6, ha="center", va="center")

        # matrix[10,2] = 99

        axes.set_xlabel("Observed Runlength")
        axes.set_ylabel("True Runlength")
        pyplot.title(base)
        pyplot.imshow(normalized_matrix)
        pyplot.show()
        pyplot.close()


def save_numpy_matrices(output_dir, filename, matrices):
    array_file_extension = ".npz"

    # ensure chromosomal directory exists
    if not os.path.exists(output_dir):
        FileManager.ensure_directory_exists(output_dir)

    output_path_prefix = os.path.join(output_dir, filename)

    output_path = output_path_prefix + array_file_extension

    # write numpy arrays
    numpy.savez_compressed(output_path, A=matrices["A"], G=matrices["G"], T=matrices["T"], C=matrices["C"])


def load_base_frequency_matrices(path):
    matrix_labels = ["A", "G", "T", "C"]
    base_frequency_matrices = dict()

    for base in matrix_labels:
        matrix = numpy.load(path)[base]
        # matrix = matrix[1:, 1:]  # trim 0 columns (for now)

        base_frequency_matrices[base] = matrix

    return base_frequency_matrices


# def generate_training_data(data_loader, batch_size, consensus_caller, output_dir, filename_suffix):
#     # datetime_string = FileManager.get_datetime_string()
#     #
#     # output_dir = os.path.join(output_dir, datetime_string)
#     # filename = "joint_distribution_" + datetime_string
#
#     n_files = len(data_loader)
#     all_training_tuples = list()
#     i = 0
#
#     print("testing n windows: ", n_files)
#
#     for b, batch in enumerate(data_loader):
#         paths, x_pileup, y_pileup, x_repeat, y_repeat, reversal = batch
#
#         # (n,h,w) shape
#         batch_size, n_channels, height, width = x_pileup.shape
#
#         for n in range(batch_size):
#             # input shape = (batch_size, n_channels, height, width)
#             # example x_pileup_n shape: (1, 5, 44, 24)
#             # example y_pileup_n shape: (1, 5, 1, 24)
#             # example x_repeat_n shape: (1, 1, 44, 24)
#             # example y_repeat_n shape: (1, 1, 1, 24)
#
#             x_pileup_n = x_pileup[n, :, :].reshape([n_channels, height, width])
#             y_pileup_n = y_pileup[n, :, :].reshape([5, 1, width])
#             x_repeat_n = x_repeat[n, :, :].reshape([1, height, width])
#             y_repeat_n = y_repeat[n, :, :].reshape([1, width])
#             reversal_n = reversal[n, :, :].reshape([1, height, width])
#
#             # print()
#             # print(x_pileup_n.shape)
#             # print(y_pileup_n.shape)
#             # print(x_repeat_n.shape)
#             # print(y_repeat_n.shape)
#
#             truths_vs_observations = get_joint_base_runlength_observations_vs_truth(x_pileup=x_pileup_n,
#                                                                                     y_pileup=y_pileup_n,
#                                                                                     x_repeat=x_repeat_n,
#                                                                                     y_repeat=y_repeat_n,
#                                                                                     reversal=reversal_n,
#                                                                                     path=paths[0])
#
#             all_training_tuples.extend(truths_vs_observations)
#
#             if i % 1 == 0:
#                 sys.stdout.write("\r " + str(round(i/n_files*100,3)) + "% completed")
#
#             if i % 10000 == 0 or i == n_files -1:
#                 filename = "training_data_" + filename_suffix + "_" + str(i)
#                 print("\nSAVING: ", os.path.join(output_dir, filename))
#                 FileManager.save_object_pickle(output_dir=output_dir, filename=filename, object=all_training_tuples)
#                 all_training_tuples = list()
#
#             i += 1
#
#     return output_dir


def generate_training_data(data_loader, batch_size, consensus_caller, output_dir, filename_suffix, gap_filterer=None):
    # datetime_string = FileManager.get_datetime_string()
    #
    # output_dir = os.path.join(output_dir, datetime_string)
    # filename = "joint_distribution_" + datetime_string

    n_files = len(data_loader)
    all_training_tuples = list()
    i = 0

    print("testing n windows: ", n_files)

    for b, batch in enumerate(data_loader):
        # sys.stdout.write("\r %.2f%% COMPLETED  " % (100*b/n_batches))

        paths, x_pileup, y_pileup, x_repeat, y_repeat, reversal = batch

        # print()
        # print("X PILEUP", x_pileup.shape)
        # print("Y PILEUP", y_pileup.shape)
        # print("X REPEAT", x_repeat.shape)
        # print("Y REPEAT", y_repeat.shape)
        # print("REVERSAL", reversal.shape)

        if gap_filterer is not None:
            try:
                batch = gap_filterer.filter_batch(batch, plot=False)
                x_pileup, y_pileup, x_repeat, y_repeat, reversal = batch

            except ValueError as e:
                print("ERROR:", e)
                print("X PILEUP", x_pileup.shape)
                print("Y PILEUP", y_pileup.shape)
                print("X REPEAT", x_repeat.shape)
                print("Y REPEAT", y_repeat.shape)
                print("REVERSAL", reversal.shape)

                continue

        # (n,h,w) shape
        batch_size, n_channels, height, width = x_pileup.shape

        for n in range(batch_size):
            # input shape = (batch_size, n_channels, height, width)
            # example x_pileup_n shape: (5, 44, 24)
            # example y_pileup_n shape: (5, 1, 24)
            # example x_repeat_n shape: (1, 44, 24)
            # example y_repeat_n shape: (1, 1, 24)

            x_pileup_n = x_pileup[n, :, :].reshape([n_channels, height, width])
            y_pileup_n = y_pileup[n, :, :].reshape([5, 1, width])
            x_repeat_n = x_repeat[n, :, :].reshape([1, height, width])
            y_repeat_n = y_repeat[n, :, :].reshape([1, width])
            reversal_n = reversal[n, :, :].reshape([1, height, width])

            truths_vs_observations = get_joint_base_runlength_observations_vs_truth(x_pileup=x_pileup_n,
                                                                                    y_pileup=y_pileup_n,
                                                                                    x_repeat=x_repeat_n,
                                                                                    y_repeat=y_repeat_n,
                                                                                    reversal=reversal_n,
                                                                                    path=paths[0])

            all_training_tuples.extend(truths_vs_observations)

            if i % 1 == 0:
                sys.stdout.write("\r " + str(round(i/n_files*100,3)) + "% completed")

            if i % 10000 == 0 or i == n_files -1:
                filename = "training_data_" + filename_suffix + "_" + str(i)
                print("\nSAVING: ", os.path.join(output_dir, filename))
                FileManager.save_object_pickle(output_dir=output_dir, filename=filename, object=all_training_tuples)
                all_training_tuples = list()

            i += 1

    return output_dir


def load_training_tuples(path, cutoff=sys.maxsize):
    paths = FileManager.get_all_file_paths_by_type(parent_directory_path=path, file_extension=".pkl")

    all_tuples = list()

    for p,path in enumerate(paths):
        with open(path, 'rb') as pickle_file:
            tuples = pickle.load(pickle_file)
            all_tuples.extend(tuples)

        print(p)
        if p == cutoff:
            break

    return all_tuples


def train_joint_model_from_tuples(tuples_path):
    training_tuples = load_training_tuples(tuples_path, cutoff=16)

    print("training tuples loaded: ", len(training_tuples))

    distribution = train_model(data=training_tuples)

    distribution_output_dir = "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/distribution/"
    distribution_filename = "distribution_" + FileManager.get_datetime_string()

    FileManager.save_object_pickle(object=distribution, filename=distribution_filename, output_dir=distribution_output_dir)


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


def run_generate_tuples_from_pileups():
    max_threads = 6

    # NC_003279.8         Caenorhabditis elegans chromosome I
    # NC_003280.10     Caenorhabditis elegans chromosome II
    # NC_003281.10     Caenorhabditis elegans chromosome III
    # NC_003282.8         Caenorhabditis elegans chromosome IV
    # NC_003283.11    Caenorhabditis elegans chromosome V
    # NC_003284.9        Caenorhabditis elegans chromosome X
    # NC_001328.1        Caenorhabditis elegans mitochondrion, complete genome

    # data_path = ["/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-15-13-10-33-0-288/NC_003279.8",
    #              "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-15-13-10-33-0-288/NC_003280.10",
    #              "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-15-13-10-33-0-288/NC_003281.10",
    #              "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-15-13-10-33-0-288/NC_003282.8",
    #              "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-15-13-10-33-0-288/NC_003283.11",
    #              "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-15-13-10-33-0-288/NC_003284.9"]

    data_path = ["/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-11-12-14-8-24-0-316/gi"]

    args = list()
    for path in data_path:
        gap_filterer = GapFilterer()

        batch_size = 1

        file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=path, file_extension=".npz")

        data_loader = DataLoader(file_paths, batch_size=batch_size, parse_batches=False)

        consensus_caller = ConsensusCaller(sequence_to_index=sequence_to_index, sequence_to_float=sequence_to_float)

        output_dir = "output/joint_runlength_base_model/" + FileManager.get_datetime_string()

        filename_suffix = path.split("/")[-1]
        print(filename_suffix)

        args.append([data_loader, batch_size, consensus_caller, output_dir, filename_suffix, gap_filterer])

        gap_filterer = None

        gc.collect()

    n_threads = min(len(args), max_threads)

    for arg in args:
        print(arg)
    print(n_threads)

    with Pool(processes=n_threads) as pool:
        pool.starmap(generate_training_data, args)


def run_joint_frequency_matrix_generation_from_pickle():
    max_runlength = 50

    path = "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_2_16_52"
    paths = FileManager.get_all_file_paths_by_type(parent_directory_path=path, file_extension=".pkl")

    frequency_matrices = defaultdict(lambda: defaultdict(lambda: numpy.zeros([max_runlength+1, max_runlength+1]))) # include 0 as a possible runlength

    cutoff = 4
    for p,path in enumerate(paths):
        with open(path, 'rb') as pickle_file:
            print(p)

            tuples = pickle.load(pickle_file)

            for tuple in tuples:
                observed = tuple[0]
                true = tuple[1]

                observed_base, observed_length = observed
                true_base, true_length = true

                # if observed_base == "C" and observed_length == 4 and true_length == 16:
                #     print(p)

                # if true_length == 0:
                #     print(observed_length, true_length)

                observed_length = min(observed_length, max_runlength)
                true_length = min(true_length, max_runlength)

                frequency_matrices[observed_base][true_base][observed_length, true_length] += 1

        if p == cutoff:
            break

    for base in ["A", "G", "T", "C"]:

        matrix = frequency_matrices[base][base].T

        print(numpy.sum(matrix))

        matrix = normalize_frequency_matrix(frequency_matrix=matrix, log_scale=True)

        # matrix[10,2] = 99

        pyplot.imshow(matrix)
        pyplot.show()
        pyplot.close()


def run_base_frequency_matrix_generation_from_tuples(filter_mismatch=False):
    max_runlength = 50

    # directories = ["/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_15_21_52_11_560358",
    #                "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_15_21_52_12_855103",
    #                "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_15_21_52_9_946240",
    #                "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_15_21_52_7_713553",
    #                "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_15_21_52_6_593646",
    #                "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_15_21_52_8_668369"]

    # directories = ["/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_40_980920/",
    #                "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_42_138805/",
    #                "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_43_176010/",
    #                "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_44_574894/",
    #                "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_46_366545/",
    #                "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_47_822627/"]

    directories = ["output/joint_runlength_base_model/2018_11_12_14_23_56_638745/"]

    all_paths = list()
    for dir in directories:
        paths = FileManager.get_all_file_paths_by_type(parent_directory_path=dir, file_extension=".pkl")
        all_paths.extend(paths)
        print(len(all_paths))

    frequency_matrices = {"A":numpy.zeros([max_runlength+1, max_runlength+1]),
                          "G":numpy.zeros([max_runlength+1, max_runlength+1]),
                          "T":numpy.zeros([max_runlength+1, max_runlength+1]),
                          "C":numpy.zeros([max_runlength+1, max_runlength+1])} # include 0 as a possible runlength

    print("loaded paths: ", len(all_paths))

    cutoff = sys.maxsize
    for p,path in enumerate(all_paths):
        with open(path, 'rb') as pickle_file:
            print(p)

            tuples = pickle.load(pickle_file)

            for tuple in tuples:
                observed_tuple = tuple[0]
                true_tuple = tuple[1]

                observed_base, observed_length = observed_tuple
                true_base, true_length = true_tuple

                observed_length = min(observed_length, max_runlength)
                true_length = min(true_length, max_runlength)

                if true_base == "-" and observed_base != "-":
                    true_base = observed_base
                    frequency_matrices[true_base][true_length, observed_length] += 1    # prefer [y,x] convention, and it plots correctly

                elif true_base == "-" and observed_base == "-":
                    for split_base in ["A", "G", "T", "C"]:
                        # add 0:0 counts to all bases
                        frequency_matrices[split_base][true_length, observed_length] += 1

                elif true_base != "-" and observed_base == "-":

                    frequency_matrices[true_base][true_length, observed_length] += 1

                else:

                    frequency_matrices[true_base][true_length, observed_length] += 1

        if p == cutoff:
            break

    for base in ["A", "G", "T", "C"]:
        print(base)
        print(frequency_matrices[base])

    # plot_frequency_matrices(frequency_matrices)

    output_dir = "/home/ryan/code/nanopore_assembly/models/parameters/"
    filename = "runlength_frequency_matrices_per_base_" + FileManager.get_datetime_string()

    print("SAVING: ", output_dir+filename)

    save_numpy_matrices(output_dir=output_dir, filename=filename, matrices=frequency_matrices)

    # frequency_matrices = load_base_frequency_matrices(os.path.join(output_dir,filename+".npz"))

    plot_frequency_matrices(frequency_matrices)


def run_batch_training_from_tuples():
    # chr_paths = ["/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_40_980920/",
    #                "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_42_138805/",
    #                "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_43_176010/",
    #                "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_44_574894/",
    #                "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_46_366545/",
    #                "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_47_822627/"]

    chr_paths = ["output/joint_runlength_base_model/2018_11_12_14_23_56_638745/"]

    trainer = JointClassifierTrainer()

    all_file_paths = list()
    for path in chr_paths:
        file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=path, file_extension=".pkl")
        all_file_paths.extend(file_paths)

    counts = trainer.get_counts_from_tuples(paths=all_file_paths)

    distribution = trainer.train_model(counts)

    distribution_output_dir = "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/distribution/"
    distribution_filename = "distribution_" + FileManager.get_datetime_string()

    print("\nSAVING: ", os.path.join(distribution_output_dir, distribution_filename))

    FileManager.save_object_pickle(object=distribution, filename=distribution_filename, output_dir=distribution_output_dir)


def plot_joint_distribution(distribution, save=False):
    base_distributions = defaultdict(lambda: numpy.zeros([max_runlength + 1, max_runlength + 1]))

    print(len(distribution))

    max_runlength = 50
    for true_base in ["A", "G", "T", "C", "-"]:
        # base_self_distribution = numpy.zeros([max_runlength + 1, max_runlength + 1])

        for observed_base in ["A", "G", "T", "C", "-"]:
            for r_x, observed_repeat in enumerate(range(0, max_runlength+1)):
                for r_y, true_repeat in enumerate(range(0, max_runlength+1)):

                    key = ((observed_base, observed_repeat),(true_base, true_repeat))

                    if key in distribution:
                        probability = distribution[key]

                        if true_base == "-" and observed_base != "-":
                            base_distributions[observed_base][r_y, r_x] += probability

                        elif true_base == "-" and observed_base == "-":
                            for split_base in ["A", "G", "T", "C"]:
                                base_distributions[split_base][r_y, r_x] += probability

                        else:
                            base_distributions[true_base][r_y,r_x] += probability

    # base_distributions["A"][25, 0] += 999999

    for base in base_distributions:
        axes = pyplot.axes()
        base_distribution = normalize_frequency_matrix(base_distributions[base], log_scale=True)
        pyplot.title(base + ":" + base + " Log probabilities")
        pyplot.imshow(numpy.log10(base_distributions[base]))

        axes.set_xlabel("Observed length")
        axes.set_ylabel("True length")
        pyplot.show()
        pyplot.close()

    if save:
        output_dir = "/home/ryan/code/nanopore_assembly/models/parameters/"
        filename = "runlength_frequency_matrices_per_base_" + FileManager.get_datetime_string()

        print("SAVING: ", output_dir + filename)

        save_numpy_matrices(output_dir=output_dir, filename=filename, matrices=base_distributions)


def plot_joint_distribution_for_all_transitions(distribution):
    base_distributions = defaultdict(lambda: defaultdict(lambda: numpy.zeros([max_runlength + 1, max_runlength + 1])))

    print(len(distribution))

    max_runlength = 30
    for true_base in ["A", "G", "T", "C", "-"]:
        # base_self_distribution = numpy.zeros([max_runlength + 1, max_runlength + 1])

        for observed_base in ["A", "G", "T", "C", "-"]:
            for r_x, observed_repeat in enumerate(range(0, max_runlength+1)):
                for r_y, true_repeat in enumerate(range(0, max_runlength+1)):

                    key = ((observed_base, observed_repeat),(true_base, true_repeat))

                    if key in distribution:
                        probability = distribution[key]

                        base_distributions[true_base][observed_base][r_y,r_x] += probability

    for true_base in ["A", "G", "T", "C", "-"]:
        # base_self_distribution = numpy.zeros([max_runlength + 1, max_runlength + 1])

        for observed_base in ["A", "G", "T", "C", "-"]:
            fig = pyplot.figure()
            axes = pyplot.axes()
            fig.set_size_inches(12, 12)

            for r_x, observed_repeat in enumerate(range(0, max_runlength + 1)):
                for r_y, true_repeat in enumerate(range(0, max_runlength + 1)):
                    probability = base_distributions[true_base][observed_base][r_y,r_x]
                    pyplot.text(r_x,r_y,str(round(numpy.log10(probability),2)), fontsize=6, ha="center", va="center")

            pyplot.title("True=" + true_base + ", Observed=" + observed_base + " log10 probabilities")
            pyplot.imshow(numpy.log10(base_distributions[true_base][observed_base]))

            axes.set_xlabel("Observed length")
            axes.set_ylabel("True length")

            pyplot.savefig("True_" + true_base + "_Observed_" + observed_base + "_probabilities.png")

            pyplot.show()
            pyplot.close()

    # base_distributions["A"][25, 0] += 999999

    # for base in base_distributions:
    #     base_distribution = normalize_frequency_matrix(base_distributions[base], log_scale=True)
    #     pyplot.title(base + ":" + base + " Log probabilities")
    #     pyplot.imshow(numpy.log10(base_distributions[base]))
    #     pyplot.show()
    #     pyplot.close()


def print_joint_distribution():
    max_runlength = 50

    chr_paths = ["/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_40_980920/",
                 "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_42_138805/",
                 "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_43_176010/",
                 "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_44_574894/",
                 "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_46_366545/",
                 "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_20_13_47_47_822627/"]

    trainer = JointClassifierTrainer()

    all_file_paths = list()
    for p,path in enumerate(chr_paths):
        file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=path, file_extension=".pkl")
        all_file_paths.extend(file_paths)

        # if p == 1:
        #     break

    counts = trainer.get_counts_from_tuples(paths=all_file_paths)

    header = '\t'.join(["observed_base", "observed_repeat", "true_base", "true_repeat", "count"])

    print(header)

    for true_base in ["A", "G", "T", "C", "-"]:
        for observed_base in  ["A", "G", "T", "C", "-"]:
            for r_x, observed_repeat in enumerate(range(0, max_runlength+1)):
                for r_y, true_repeat in enumerate(range(0, max_runlength+1)):
                    key = ((observed_base, observed_repeat),(true_base, true_repeat))
                    count = counts[key]

                    line = '\t'.join([observed_base, str(observed_repeat), true_base, str(true_repeat), str(count)])

                    print(line)


def main():
    run_base_frequency_matrix_generation_from_tuples(filter_mismatch=False)
    # run_generate_tuples_from_pileups()
    # run_batch_training_from_tuples()

    # Joint dirichlet distribution
    # path = "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/distribution/distribution_2018_10_21_13_24_32_964551.pkl" # c elegans filtered
    path = "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/distribution/distribution_2018_11_12_14_47_47_46658.pkl" # e coli filtered

    # with open(path, 'rb') as pickle_file:
    #     distribution = pickle.load(pickle_file)

    plot_joint_distribution_for_all_transitions(distribution)

    # # frequencies of all pairwise observations (POA filtered by RNN)
    # path = "/home/ryan/code/nanopore_assembly/models/parameters/joint_frequencies_chr1-6_celegans_filtered_POA.txt"

    # distribution = dict()
    # with open(path, "r") as file:
    #     for l,line in enumerate(file):
    #         if l > 0:
    #             line = line.strip().split("\t")
    #             observed_base, observed_repeat, true_base, true_repeat, count = line
    #
    #             observed_repeat = int(observed_repeat)
    #             true_repeat = int(true_repeat)
    #             count = int(count)
    #
    #             key = ((observed_base, observed_repeat), (true_base, true_repeat))
    #
    #             # print(key, count)
    #
    #             distribution[key] = count

    # plot_joint_distribution(distribution, save=True)

    # print_joint_distribution()

    # path = "/home/ryan/code/nanopore_assembly/models/parameters/runlength_frequency_matrices_per_base_2018-9-20-14-21-6.npz"
    #
    # frequency_matrices = load_base_frequency_matrices(path)
    # plot_frequency_matrices(frequency_matrices)


if __name__ == "__main__":
    main()