from models.JointClassifierTrainer import train_model, JointClassifierTrainer
from modules.pileup_utils import sequence_to_float, sequence_to_index, index_to_sequence, get_joint_base_runlength_observations_vs_truth
from modules.ConsensusCaller import ConsensusCaller
from handlers.DataLoaderRunlength import DataLoader
from handlers.FileManager import FileManager
from collections import defaultdict
from matplotlib import pyplot
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
import pickle
import numpy
import sys
import os


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
        matrix = matrix[1:, 1:]  # trim 0 columns (for now)

        base_frequency_matrices[base] = matrix

    return base_frequency_matrices


def generate_training_data(data_loader, batch_size, consensus_caller, output_dir, filename_suffix):
    # datetime_string = FileManager.get_datetime_string()
    #
    # output_dir = os.path.join(output_dir, datetime_string)
    # filename = "joint_distribution_" + datetime_string

    n_files = len(data_loader)
    all_training_tuples = list()
    i = 0

    print("testing n windows: ", n_files)

    for b, batch in enumerate(data_loader):
        paths, x_pileup, y_pileup, x_repeat, y_repeat, reversal = batch

        # (n,h,w) shape
        batch_size, n_channels, height, width = x_pileup.shape

        for n in range(batch_size):
            # input shape = (batch_size, n_channels, height, width)
            # example x_pileup_n shape: (1, 5, 44, 24)
            # example y_pileup_n shape: (1, 5, 1, 24)
            # example x_repeat_n shape: (1, 1, 44, 24)
            # example y_repeat_n shape: (1, 1, 1, 24)

            x_pileup_n = x_pileup[n, :, :].reshape([n_channels, height, width])
            y_pileup_n = y_pileup[n, :, :].reshape([5, 1, width])
            x_repeat_n = x_repeat[n, :, :].reshape([1, height, width])
            y_repeat_n = y_repeat[n, :, :].reshape([1, width])
            reversal_n = reversal[n, :, :].reshape([1, height, width])

            # print()
            # print(x_pileup_n.shape)
            # print(y_pileup_n.shape)
            # print(x_repeat_n.shape)
            # print(y_repeat_n.shape)

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


def run_train_from_pileups():
    max_threads = 30

    # NC_003279.8         Caenorhabditis elegans chromosome I
    # NC_003280.10     Caenorhabditis elegans chromosome II
    # NC_003281.10     Caenorhabditis elegans chromosome III
    # NC_003282.8         Caenorhabditis elegans chromosome IV
    # NC_003283.11    Caenorhabditis elegans chromosome V
    # NC_003284.9        Caenorhabditis elegans chromosome X
    # NC_001328.1        Caenorhabditis elegans mitochondrion, complete genome

    data_path = ["/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-2-10-43-22-1-275/NC_003279.8",    # one hot, reversal encoded, chr1 c elegans full
                "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-2-10-43-22-1-275/NC_003280.10",    # one hot, reversal encoded, chr2 c elegans full
                "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-2-10-43-22-1-275/NC_003281.10",    # one hot, reversal encoded, chr3 c elegans full
                "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-2-10-43-22-1-275/NC_003282.8",     # one hot, reversal encoded, chr4 c elegans full
                "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-2-10-43-22-1-275/NC_003283.11",    # one hot, reversal encoded, chr5 c elegans full
                "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-2-10-43-22-1-275/NC_003284.9"]     # one hot, reversal encoded, chrX c elegans full

    args = list()
    for path in data_path:
        batch_size = 1

        file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=path, file_extension=".npz")

        data_loader = DataLoader(file_paths, batch_size=batch_size, parse_batches=False)

        consensus_caller = ConsensusCaller(sequence_to_index=sequence_to_index, sequence_to_float=sequence_to_float)

        output_dir = "output/joint_runlength_base_model/" + FileManager.get_datetime_string()

        filename_suffix = path.split("/")[-1]
        print(filename_suffix)

        args.append([data_loader, batch_size, consensus_caller, output_dir, filename_suffix])

    n_threads = min(len(args), max_threads)
    with Pool(processes=n_threads) as pool:
        pool.starmap(generate_training_data, args)


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


def run_base_frequency_matrix_generation_from_pickle(filter_mismatch=False):
    max_runlength = 50

    path = "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_2_16_52"
    paths = FileManager.get_all_file_paths_by_type(parent_directory_path=path, file_extension=".pkl")

    frequency_matrices = defaultdict(lambda: numpy.zeros([max_runlength+1, max_runlength+1])) # include 0 as a possible runlength

    cutoff = 8
    for p,path in enumerate(paths):
        with open(path, 'rb') as pickle_file:
            print(p)

            tuples = pickle.load(pickle_file)

            for tuple in tuples:
                observed = tuple[0]
                true = tuple[1]

                observed_base, observed_length = observed
                true_base, true_length = true

                if filter_mismatch:
                    if observed_base != true_base:
                        continue

                observed_length = min(observed_length, max_runlength)
                true_length = min(true_length, max_runlength)

                frequency_matrices[true_base][true_length, observed_length] += 1    # i prefer [y,x] convention, and it plots correctly

        if p == cutoff:
            break

    plot_frequency_matrices(frequency_matrices)

    output_dir = "/home/ryan/code/nanopore_assembly/models/parameters/"
    filename = "runlength_frequency_matrices_per_base_" + FileManager.get_datetime_string()

    save_numpy_matrices(output_dir=output_dir, filename=filename, matrices=frequency_matrices)
    # frequency_matrices = load_base_frequency_matrices(os.path.join(output_dir,filename+".npz"))
    #
    # plot_frequency_matrices(frequency_matrices)


def run_batch_training_from_tuples():
    chr_paths = ["/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_9_15_3_33_156698",
                 "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_9_15_3_34_558893",
                 "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_9_15_3_35_795050",
                 "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_9_15_3_37_409932",
                 "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_9_15_3_39_376950",
                 "/home/ryan/code/nanopore_assembly/output/joint_runlength_base_model/2018_10_9_15_3_41_28347"]

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
    # run_base_frequency_matrix_generation_from_pickle(filter_mismatch=True)
    # run_train_from_pileups()
    run_batch_training_from_tuples()

    # path = "/home/ryan/code/nanopore_assembly/models/parameters/runlength_frequency_matrices_per_base_2018-9-20-14-21-6.npz"
    #
    # frequency_matrices = load_base_frequency_matrices(path)
    # plot_frequency_matrices(frequency_matrices)
