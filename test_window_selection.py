from handlers.FileManager import FileManager
from handlers.FastaHandler import FastaHandler
from modules.IntervalTree import IntervalTree
from generate_spoa_pileups_from_bam import *
from collections import defaultdict
from modules.window_selection_utils import merge_windows
from matplotlib import pyplot
import pickle
import numpy


def find_repeats(sequence, repeat_threshold):
    """
    For a sequence, generate a dictionary of characters:observed_repeats, where observed_repeats is a list of n
    repeats observed for all sequential stretches of that character in the sequence
    :param sequences:
    :return:
    """
    # character_counts = defaultdict(list)
    repeat_positions = dict()
    current_character = None
    current_count = 0

    for c,character in enumerate(sequence):
        if character != current_character:
            # character_counts[character].append(current_count)

            if current_count > repeat_threshold:
                position = c-current_count

                # print(position, current_count)
                # print(type(position), type(current_count))
                repeat_positions[position] = current_count

            current_count = 0   # reset
        else:
            current_count += 1

        current_character = character

        # if c > 100000:
        #     break

    return repeat_positions


def count_total_window_coverage(windows):
    total = 0
    for window in windows:
        window_length = window[1] - window[0]

        total += window_length

    return total


def plot_split_ratios_per_length(split_counts_per_length):
    max_length = max(split_counts_per_length.keys())

    matrix = numpy.zeros([1,max_length+1], dtype=numpy.float32)

    axes = pyplot.axes()
    for length in split_counts_per_length:
        split_count = split_counts_per_length[length]["split"]
        unsplit_count = split_counts_per_length[length]["unsplit"]

        ratio = split_count/(split_count+unsplit_count)

        matrix[0,length] = ratio
        pyplot.text(length, 0, str(round(ratio*100,1)), ha="center", va="center")

    print(matrix)
    pyplot.imshow(matrix)

    axes.set_xlabel("length of repeat")
    axes.set_ylabel("split likelihood")

    pyplot.show()
    pyplot.close()


def plot_pileups_for_split_repeats(split_repeat_windows, bam_file_path, reference_file_path, chromosome_name):
    output_dir = "test"

    for item in split_repeat_windows:
        start = item["start"]
        stop = item["stop"]
        windows = item["windows"]

        if stop - start > 4:
            print("start:", start, "stop:", stop, "windows", windows)
            for window in windows:
                generate_window_run_length_encoding(bam_file_path=bam_file_path,
                                                    reference_file_path=reference_file_path,
                                                    chromosome_name=chromosome_name,
                                                    window=window,
                                                    output_dir=output_dir,
                                                    sort_sequences_by_length=False,
                                                    reverse_sort=False,
                                                    two_pass=True,
                                                    plot_results=True,
                                                    print_results=True,
                                                    save_data=False)


def locate_repeats_in_anchored_windows(windows, repeat_positions):
    split_counts_per_length = defaultdict(lambda:{"split":0, "unsplit":0})
    split_repeat_windows = list()
    unsplit_repeat_windows = list()

    window_interval_tree = IntervalTree(windows)

    split_repeats = 0
    normal_repeats = 0
    lost_repeats = 0

    for item in repeat_positions.items():
        position, length = item

        # print(position, length)
        # print(type(position), type(length))

        matching_windows = window_interval_tree.find(start=position, stop=position+length-1)

        if len(matching_windows) > 1:
            split_repeats += 1
            split_counts_per_length[length]["split"] += 1
            split_repeat_windows.append({"start": position, "stop": position + length, "windows": matching_windows})

        elif len(matching_windows) == 1:
            normal_repeats += 1
            split_counts_per_length[length]["unsplit"] += 1

            if length > 9:
                unsplit_repeat_windows.append({"start": position, "stop": position + length, "windows": matching_windows})

        elif len(matching_windows) == 0:
            lost_repeats += 1

        else:
            print("ERROR unaccounted for case:", matching_windows)

    print(split_repeats)
    print(normal_repeats)
    print(lost_repeats)
    print(split_repeats/(normal_repeats+split_repeats))

    return split_counts_per_length, split_repeat_windows, unsplit_repeat_windows


def main():
    # ---- Nanopore GUPPY - C ELEGANS - (dev machine) -------------------------
    bam_file_path = "/home/ryan/data/Nanopore/celegans/all_chips_20k_Boreal_minimap2.sorted.bam"
    reference_file_path = "/home/ryan/data/Nanopore/celegans/GCF_000002985.6_WBcel235_genomic.fasta"
    # -------------------------------------------------------------------------

    fasta_handler = FastaHandler(reference_file_path)

    # chromosomal_window_path = "output/window_selection/NC_003279.8_0_15072434_2018_10_1_20_1"   # kernel method
    chromosomal_window_path = "/home/ryan/code/nanopore_assembly/output/window_selection/NC_003279.8_0_15072434_2018_10_12_10_58_56_199382"     # transition method
    chromosome_name = "NC_003279.8"

    chromosome_length = fasta_handler.get_chr_sequence_length(chromosome_name)

    reference_sequence = fasta_handler.get_sequence(chromosome_name=chromosome_name, start=0, stop=chromosome_length)

    windows = load_windows(chromosomal_window_path)

    long_repeat_positions = find_repeats(sequence=reference_sequence, repeat_threshold=1)

    split_counts_per_length, split_repeat_windows, unsplit_repeat_windows = \
        locate_repeats_in_anchored_windows(windows=windows, repeat_positions=long_repeat_positions)

    plot_split_ratios_per_length(split_counts_per_length)
    plot_pileups_for_split_repeats(split_repeat_windows=split_repeat_windows,
                                   bam_file_path=bam_file_path,
                                   reference_file_path=reference_file_path,
                                   chromosome_name=chromosome_name)


if __name__ == "__main__":
    main()
