from modules.AlignedSegmentGrabber import SegmentGrabber
from modules.IntervalTree import IntervalTree
from handlers.VcfHandler import VCFFileProcessor
from handlers.FastaHandler import FastaHandler
from handlers.BamHandler import BamHandler
from handlers.TsvHandler import TsvHandler
from handlers.FastaWriter import FastaWriter
from select_windows import WINDOW_SIZE, CDF_STEP_SIZE
from modules.window_selection_utils import merge_windows
from modules.pileup_utils import *
from modules.alignment_utils import *
from multiprocessing import Pool
import multiprocessing
import random
from matplotlib import pyplot
from datetime import datetime
from tqdm import tqdm
import pickle
import math
import os.path


def print_collapsed_segments(sequences, character_counts):
    for i in range(len(sequences)):
        sequence = sequences[i]
        character_count = ''.join(map(str,character_counts[i]))

        print(i)
        print(sequence)
        print(character_count)


def print_segments(ref_sequence, sequences):
    print(ref_sequence)
    print("-" * len(ref_sequence))
    for sequence in sequences:
        print(sequence)
    print()


def get_current_timestamp():
    datetime_string = '-'.join(list(map(str, datetime.now().timetuple()))[:-1])

    return datetime_string


def get_alignments_by_sequence(alignments, sequence):
    """
    iterate through alignment tuples to find all alignments with the specified sequence
    :param alignments:
    :param sequence:
    :return:
    """
    query_alignments = list()

    for a, alignment in enumerate(alignments):
        current_sequence = alignment[1].replace("-",'')

        if current_sequence == sequence:
            query_alignments.append(alignment)

    if len(query_alignments) == 0:
        raise KeyError("ERROR: query sequence not found in alignments")

    return query_alignments


def get_aligned_segments(fasta_handler, bam_handler, chromosome_name, pileup_start, pileup_end, include_ref=False, exclude_loose_ends=True):
    """
    Get read segments from a pair of coordinates given that each read has an aligned match at the start and end
    coordinate
    :param fasta_handler:
    :param bam_handler:
    :param chromosome_name:
    :param pileup_start:
    :param pileup_end:
    :return:
    """
    ref_sequence = fasta_handler.get_sequence(chromosome_name=chromosome_name,
                                              start=pileup_start,
                                              stop=pileup_end + 1)

    reads = bam_handler.get_reads(chromosome_name=chromosome_name,
                                  start=pileup_start,
                                  stop=pileup_end)

    # for read in reads:
    #     print(read.query_name[:10], read.query_length)

    segment_grabber = SegmentGrabber(chromosome_name=chromosome_name,
                                     start_position=pileup_start,
                                     end_position=pileup_end,
                                     ref_sequence=ref_sequence,
                                     reads=reads,
                                     exclude_loose_ends=exclude_loose_ends)

    # if a reference sequence is intended to be added to the pileup, then leave a space for it
    if include_ref:
        segment_grabber.max_coverage -= 1

    sequence_dictionary, reverse_status_dictionary = segment_grabber.get_read_segments()

    if len(sequence_dictionary.keys()) == 0:
        print("\nWARNING: No reads found at position:", pileup_start, pileup_end)
        ref_sequence = None
        read_ids = None
        sequences = None
        reversal_statuses = None
    else:
        read_ids = list()
        sequences = list()
        reversal_statuses = list()

        for read_id in sequence_dictionary:
            sequence = sequence_dictionary[read_id]
            reverse_status = reverse_status_dictionary[read_id]

            sequences.append(sequence)
            reversal_statuses.append(reverse_status)

    return ref_sequence, read_ids, sequences, reversal_statuses


def build_chromosomal_interval_trees(confident_bed_path):
    """
    Produce a dictionary of intervals trees, with one tree per chromosome
    :param confident_bed_path: Path to confident bed file
    :return: trees_chromosomal
    """
    # create an object for tsv file handling
    tsv_handler_reference = TsvHandler(tsv_file_path=confident_bed_path)

    # create intervals based on chromosome
    intervals_chromosomal = tsv_handler_reference.get_bed_intervals_by_chromosome(start_offset=1, universal_offset=-1)

    # create a dictionary to get all chromosomal trees
    trees_chromosomal = dict()

    # for each chromosome extract the tree and add it to the dictionary
    for chromosome_name in intervals_chromosomal:
        intervals = intervals_chromosomal[chromosome_name]
        tree = IntervalTree(intervals)

        trees_chromosomal[chromosome_name] = tree

    # return the dictionary containing all the trees
    return trees_chromosomal, intervals_chromosomal


def test_chunks_vs_interval(interval, chunks, chunk_size):
    """
    test method for chunking intervals
    :param interval:
    :param chunks:
    :param chunk_size:
    :return:
    """
    chunk_steps = list()
    interval_steps = list()

    for chunk in chunks:
        last_i = None
        for i in range(chunk[0], chunk[1]):

            length = chunk[1] - chunk[0]
            if length > chunk_size:
                print("WARNING: INCORRECT CHUNK LENGTH:", length, chunk, interval)

            chunk_steps.append(i)

            if i == last_i:
                print(chunk, i)

            last_i = i

    for i in range(interval[0], interval[1]):
        interval_steps.append(i)

    if chunk_steps != interval_steps:
        print("WARNING UNEQUAL STEPS", interval)
        print(len(chunk_steps), len(interval_steps))


def get_chunked_intervals(chromosomal_intervals, chunk_size, test=False):
    """
    For a list of chromosomal confident intervals, generate a list of smaller intervals with len chunk size
    :param chromosomal_intervals:
    :param chunk_size:
    :param test:
    :return:
    """
    chromosomal_chunked_intervals = dict()

    for chromosome_name in chromosomal_intervals:
        chunked_intervals = list()

        for interval in chromosomal_intervals[chromosome_name]:
            length = interval[1] - interval[0]

            if length > chunk_size:
                chunks = chunk_interval(interval=interval, length=length, chunk_size=chunk_size)
                chunked_intervals.extend(chunks)

                if test:
                    test_chunks_vs_interval(interval=interval, chunks=chunks, chunk_size=chunk_size)

            else:
                chunked_intervals.append(interval)

        chromosomal_chunked_intervals[chromosome_name] = chunked_intervals

    return chromosomal_chunked_intervals


def chunk_interval(interval, length, chunk_size):
    """
    Split an individual interval into intervals of length chunk_size
    :param interval:
    :param length:
    :param chunk_size:
    :return:
    """
    chunks = list()

    n_chunks = math.ceil(length/chunk_size)

    start = interval[0]
    for i in range(n_chunks):
        end = start + chunk_size

        if end > interval[1]:
            end = interval[1]

        chunk = [start, end]

        chunks.append(chunk)

        start = end

    return chunks


def get_variant_windows(vcf_path, chromosome_name, start_position, end_position):
    """
    Get the positions at which variants are found, in the vcf
    :param vcf_path:
    :param chromosome_name:
    :param start_position:
    :param end_position:
    :return:
    """
    vcf_handler = VCFFileProcessor(vcf_path)

    vcf_handler.populate_dictionary(contig=chromosome_name,
                                    start_pos=start_position,
                                    end_pos=end_position + 1)

    positional_variants = vcf_handler.get_variant_dictionary()
    positions = positional_variants.keys()

    windows = list()
    for p,position in enumerate(positions):
        pileup_start = position - 5
        pileup_end = position + 5      # add random variation here
        windows.append([pileup_start, pileup_end])

    return windows


def filter_windows(chromosomal_chunked_intervals, window_size, variant_positions):
    """
    For a list of intervals (windows) in each chromosome, filter by size and the presence of a variant in the window.
    :param chromosomal_chunked_intervals:
    :param size:
    :param variant_positions:
    :return:
    """
    variant_intervals = list()
    for position in variant_positions:
        interval = [position, position]
        variant_intervals.append(interval)

    variant_interval_tree = IntervalTree(variant_intervals)

    chromosomal_filtered_windows = dict()
    for chromosome_name in chromosomal_chunked_intervals:
        intervals = list()
        for interval in chromosomal_chunked_intervals[chromosome_name]:
            length = interval[1] - interval[0]

            contains_variant = interval in variant_interval_tree
            correct_size = length == window_size

            if not contains_variant and correct_size:
                intervals.append(interval)

        chromosomal_filtered_windows[chromosome_name] = intervals

        print(chromosome_name)
        print("# windows before filtering:", len(chromosomal_chunked_intervals[chromosome_name]))
        print("# windows after filtering:", len(chromosomal_filtered_windows[chromosome_name]))

    return chromosomal_filtered_windows


def get_non_variant_windows(vcf_path, bed_path, chromosome_name, start_position, end_position):
    """
    :param vcf_path:
    :param bed_path:
    :param chromosome_name:
    :param start_position:
    :param end_position:
    :return:
    """
    chunk_size = 20

    print("Populating variant dictionary...")
    vcf_handler = VCFFileProcessor(vcf_path)
    vcf_handler.populate_dictionary(contig=chromosome_name,
                                    start_pos=start_position,
                                    end_pos=end_position + 1)

    positional_variants = vcf_handler.get_variant_dictionary()
    variant_positions = positional_variants.keys()

    print("Reading confident intervals...")
    tsv_handler = TsvHandler(bed_path)
    chromosomal_intervals = tsv_handler.get_bed_intervals_by_chromosome(start_offset=1, universal_offset=-1)
    chromosomal_intervals = {chromosome_name:chromosomal_intervals[chromosome_name]}

    print("Chunking intervals...")
    chromosomal_chunked_intervals = get_chunked_intervals(chromosomal_intervals=chromosomal_intervals, chunk_size=chunk_size)

    print("Filtering windows...")
    filtered_windows = filter_windows(chromosomal_chunked_intervals=chromosomal_chunked_intervals,
                                      window_size=chunk_size,
                                      variant_positions=variant_positions)

    return filtered_windows


def load_windows(path):
    window_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=path, file_extension=".pkl")

    positional_window_paths = list()

    for window_path in window_paths:
        start = int(window_path.split("/")[-1].split("_")[0])
        positional_window_paths.append([start, window_path])

    positional_window_paths = sorted(positional_window_paths, key=lambda x: x[0])

    merged_windows = list()
    # w = 0
    for item in positional_window_paths:
        path = item[1]

        with open(path, 'rb') as pickle_file:
            window_list = pickle.load(pickle_file)

            length = len(merged_windows)

            if length > 0:
                # print(window_list[0])
                # print(merged_windows[-1])

                merged_windows = merge_windows(windows_a=merged_windows, windows_b=window_list, max_size=CDF_STEP_SIZE)

                # print(merged_windows[length-2:length+2])
            else:
                merged_windows = window_list

            # if w == 2:
            #     exit()
            # w+=1

    return merged_windows

# def load_windows(path):
#     window_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=path, file_extension=".pkl")
#     windows = list()
#
#     positional_window_paths = list()
#
#     for window_path in window_paths:
#         start = int(window_path.split("/")[-1].split("_")[0])
#         windows.append([start, window_path])


# def generate_data(bam_file_path, reference_file_path, vcf_path, bed_path, chromosome_name, start_position, end_position, output_dir, generate_from_vcf=False, runlength=False):
#     """
#     Generate pileups from BAM data, and collapse sequences to have no explicitly repeated characters. Additionally
#     encode a repeat channel that describes the number of repeats observed per base.
#     :param bam_file_path:
#     :param reference_file_path:
#     :param vcf_path:
#     :param chromosome_name:
#     :param start_position:
#     :param end_position:
#     :return:
#     """
#     if runlength:
#         encode_window = generate_window_run_length_encoding
#     else:
#         encode_window = generate_window_encoding
#
#     if generate_from_vcf:
#         chromosomal_windows = get_variant_windows(vcf_path=vcf_path,
#                                                   chromosome_name=chromosome_name,
#                                                   start_position=start_position,
#                                                   end_position=end_position)
#
#     else:
#         chromosomal_windows = get_non_variant_windows(vcf_path=vcf_path,
#                                                       bed_path=bed_path,
#                                                       chromosome_name=chromosome_name,
#                                                       start_position=start_position,
#                                                       end_position=end_position)
#
#     for chromosome_name in chromosomal_windows:
#         for w,window in enumerate(chromosomal_windows[chromosome_name]):
#             pileup_start = window[0]
#             pileup_end = window[1]      # add random variation here
#
#             print(pileup_start, pileup_end)
#
#             encode_window(bam_file_path=bam_file_path,
#                           reference_file_path=reference_file_path,
#                           chromosome_name=chromosome_name,
#                           window=window,
#                           output_dir=output_dir,
#                           save_data=True,
#                           print_results=False,
#                           plot_results=False)


def generate_window_fasta(bam_file_path, reference_file_path, chromosome_name, window, output_dir, exclude_loose_ends=True):
    """
    Run the pileup generator for a single specified window
    :param bam_file_path:
    :param reference_file_path:
    :param chromosome_name:
    :param window:
    :return:
    """
    bam_handler = BamHandler(bam_file_path)
    fasta_handler = FastaHandler(reference_file_path)

    pileup_start = window[0]
    pileup_end = window[1]      # add random variation here ?

    reads_found = True

    ref_sequence, read_ids, sequences, reversal_statuses = get_aligned_segments(fasta_handler=fasta_handler,
                                                                                bam_handler=bam_handler,
                                                                                chromosome_name=chromosome_name,
                                                                                pileup_start=pileup_start,
                                                                                pileup_end=pileup_end,
                                                                                include_ref=True,
                                                                                exclude_loose_ends=exclude_loose_ends)

    if sequences is not None:
        for sequence in sequences:
            print(len(sequence))

        print(len(ref_sequence))

        FileManager.ensure_directory_exists(output_dir)
        sequences_output_filename = '_'.join([chromosome_name, str(window[0]), str(window[1])]) + ".fasta"
        sequences_output_path = os.path.join(output_dir,sequences_output_filename)
        fasta_writer = FastaWriter(sequences_output_path)
        fasta_writer.write_sequences(sequences)

        ref_output_filename = '_'.join([chromosome_name, str(window[0]), str(window[1]), "ref"]) + ".fasta"
        ref_output_path = os.path.join(output_dir,ref_output_filename)
        fasta_writer = FastaWriter(ref_output_path)
        fasta_writer.write_sequences([ref_sequence])

        # print("saving sequences as fasta: ", sequences_output_path, ref_output_path)

    else:
        reads_found = False

    return reads_found


def generate_window_encoding(bam_file_path, reference_file_path, chromosome_name, window, output_dir,
                                        sort_sequences_by_length=False, reverse_sort=False, two_pass=False,
                                        save_data=True, print_results=False, plot_results=False, counter=None,
                                        n_chunks=None, save_to_fasta=False):
    """
    Run the pileup generator for a single specified window
    :param bam_file_path:
    :param reference_file_path:
    :param chromosome_name:
    :param window:
    :return:
    """
    bam_handler = BamHandler(bam_file_path)
    fasta_handler = FastaHandler(reference_file_path)

    pileup_start = window[0]
    pileup_end = window[1]      # add random variation here ?

    ref_sequence, read_ids, sequences, reversal_statuses = get_aligned_segments(fasta_handler=fasta_handler,
                                                                                bam_handler=bam_handler,
                                                                                chromosome_name=chromosome_name,
                                                                                pileup_start=pileup_start,
                                                                                pileup_end=pileup_end,
                                                                                include_ref=True)

    if save_to_fasta:
        FileManager.ensure_directory_exists(output_dir)
        sequences_output_filename = '_'.join([chromosome_name, str(window[0]), str(window[1])]) + ".fasta"
        sequences_output_path = os.path.join(output_dir,sequences_output_filename)
        fasta_writer = FastaWriter(sequences_output_path)
        fasta_writer.write_sequences(sequences)

        ref_output_filename = '_'.join([chromosome_name, str(window[0]), str(window[1]), "ref"]) + ".fasta"
        ref_output_path = os.path.join(output_dir,ref_output_filename)
        fasta_writer = FastaWriter(ref_output_path)
        fasta_writer.write_sequences([ref_sequence])

        print("saving sequences as fasta: ", sequences_output_path, ref_output_path)

    if sequences is None:
        return

    if sort_sequences_by_length:
        for sequence in sequences:
            print(sequence)
        print()

        sequences = sorted(sequences, key=lambda x: len(x), reverse=reverse_sort)
        for sequence in sequences:
            print(sequence)

        print("ref",ref_sequence)

    # ref_sequence = ref_sequence[0]

    # print(ref_sequence)

    alignments, ref_alignment = get_spoa_alignment(sequences=sequences, ref_sequence=ref_sequence, two_pass=two_pass)

    pileup_matrix = convert_alignments_to_one_hot_tensor(alignments, fixed_coverage=False)

    reference_matrix = convert_alignments_to_one_hot_tensor(ref_alignment, fixed_coverage=False)

    reversal_matrix = convert_reversal_statuses_to_integer_matrix(reverse_statuses=reversal_statuses,
                                                                  pileup_matrix=pileup_matrix)

    pileup_repeat_matrix = numpy.ones(pileup_matrix.shape)
    reference_repeat_matrix = numpy.ones(reference_matrix.shape)

    if plot_results:
        n_channels, height, width = pileup_matrix.shape

        x_pileup = pileup_matrix.reshape([n_channels, height, width])
        y_pileup = reference_matrix.reshape([5, 1, width])
        x_repeat = numpy.ones([height, width])
        y_repeat = numpy.ones([1, width])
        reversal = reversal_matrix.reshape([height, width])

        print(x_pileup.shape)
        print(y_pileup.shape)
        print(x_repeat.shape)
        print(y_repeat.shape)
        print(reversal.shape)

        x_pileup_flat = flatten_one_hot_tensor(x_pileup)
        y_pileup_flat = flatten_one_hot_tensor(y_pileup)
        plot_runlength_prediction_stranded(x_pileup=x_pileup_flat,
                                           x_repeat=x_repeat,
                                           y_pileup=y_pileup_flat,
                                           y_repeat=y_repeat,
                                           reversal=reversal,
                                           show_reversal=False,
                                           label=True)

    if print_results:
        print_segments(ref_sequence, sequences)

        for a, alignstring in enumerate(alignments):
            print("{0:15s} {1:s}".format(str(a), alignstring))

        for alignstring in ref_alignment:
            print("{0:15s} {1:s}".format("ref", alignstring))

    if ref_alignment[0].replace("-",'') != ref_sequence:
        print("Aligned reference does not match true reference at [%d,%d]"%(pileup_start,pileup_end))
        print("unaligned:\t",ref_sequence)
        print("aligned:\t",ref_alignment[0][1].replace("-",''))

    elif save_data:
        save_run_length_training_data(output_dir=output_dir,
                                      pileup_matrix=pileup_matrix,
                                      reference_matrix=reference_matrix,
                                      pileup_repeat_matrix=pileup_repeat_matrix,
                                      reference_repeat_matrix=reference_repeat_matrix,
                                      reversal_matrix=reversal_matrix,
                                      chromosome_name=chromosome_name,
                                      start=pileup_start)

    if counter is not None:
        counter.value += 1

        sys.stdout.write('\r' + "%.2f%% Completed" % (100 * counter.value / n_chunks))


def generate_window_run_length_encoding(bam_file_path, reference_file_path, chromosome_name, window, output_dir,
                                        sort_sequences_by_length=False, reverse_sort=False, two_pass=False,
                                        save_data=True, print_results=False, plot_results=False, counter=None,
                                        n_chunks=None, save_to_fasta=False):
    """
    Run the pileup generator for a single specified window
    :param bam_file_path:
    :param reference_file_path:
    :param chromosome_name:
    :param window:
    :return:
    """
    bam_handler = BamHandler(bam_file_path)
    fasta_handler = FastaHandler(reference_file_path)

    pileup_start = window[0]
    pileup_end = window[1]      # add random variation here ?

    ref_sequence, read_ids, sequences, reversal_statuses = get_aligned_segments(fasta_handler=fasta_handler,
                                                                                bam_handler=bam_handler,
                                                                                chromosome_name=chromosome_name,
                                                                                pileup_start=pileup_start,
                                                                                pileup_end=pileup_end,
                                                                                include_ref=True)

    if save_to_fasta:
        FileManager.ensure_directory_exists(output_dir)
        sequences_output_filename = '_'.join([chromosome_name, str(window[0]), str(window[1])]) + ".fasta"
        sequences_output_path = os.path.join(output_dir,sequences_output_filename)
        fasta_writer = FastaWriter(sequences_output_path)
        fasta_writer.write_sequences(sequences)

        ref_output_filename = '_'.join([chromosome_name, str(window[0]), str(window[1]), "ref"]) + ".fasta"
        ref_output_path = os.path.join(output_dir,ref_output_filename)
        fasta_writer = FastaWriter(ref_output_path)
        fasta_writer.write_sequences([ref_sequence])

        print("saving sequences as fasta: ", sequences_output_path, ref_output_path)

    if sequences is None:
        return

    if sort_sequences_by_length:
        for sequence in sequences:
            print(sequence)
        print()

        sequences = sorted(sequences, key=lambda x: len(x), reverse=reverse_sort)
        for sequence in sequences:
            print(sequence)

        print("ref",ref_sequence)

    sequences, repeats = collapse_repeats(sequences)
    ref_sequence, ref_repeats = collapse_repeats([ref_sequence])
    ref_sequence = ref_sequence[0]

    alignments, ref_alignment = get_spoa_alignment(sequences=sequences, ref_sequence=ref_sequence, two_pass=two_pass)

    pileup_matrix, pileup_repeat_matrix = convert_collapsed_alignments_to_one_hot_tensor(alignments,
                                                                                         repeats,
                                                                                         fixed_coverage=False)

    reference_matrix, reference_repeat_matrix = convert_collapsed_alignments_to_one_hot_tensor(ref_alignment,
                                                                                               ref_repeats,
                                                                                               fixed_coverage=False)

    reversal_matrix = convert_reversal_statuses_to_integer_matrix(reverse_statuses=reversal_statuses,
                                                                  pileup_matrix=pileup_matrix)

    if plot_results:
        n_channels, height, width = pileup_matrix.shape

        x_pileup = pileup_matrix.reshape([n_channels, height, width])
        y_pileup = reference_matrix.reshape([5, 1, width])
        x_repeat = pileup_repeat_matrix.reshape([height, width])
        y_repeat = reference_repeat_matrix.reshape([1, width])
        reversal = reversal_matrix.reshape([height, width])

        print(x_pileup.shape)
        print(y_pileup.shape)
        print(x_repeat.shape)
        print(y_repeat.shape)
        print(reversal.shape)

        x_pileup_flat = flatten_one_hot_tensor(x_pileup)
        y_pileup_flat = flatten_one_hot_tensor(y_pileup)
        plot_runlength_prediction_stranded(x_pileup=x_pileup_flat,
                                           x_repeat=x_repeat,
                                           y_pileup=y_pileup_flat,
                                           y_repeat=y_repeat,
                                           reversal=reversal,
                                           show_reversal=False,
                                           label=True)

    if print_results:
        print_segments(ref_sequence, sequences)

        for a, alignstring in enumerate(alignments):
            print("{0:15s} {1:s}".format(str(a), alignstring))

        for alignstring in ref_alignment:
            print("{0:15s} {1:s}".format("ref", alignstring))

    if ref_alignment[0].replace("-",'') != ref_sequence:
        print("Aligned reference does not match true reference at [%d,%d]"%(pileup_start,pileup_end))
        print("unaligned:\t",ref_sequence)
        print("aligned:\t",ref_alignment[0][1].replace("-",''))

    elif save_data:
        save_run_length_training_data(output_dir=output_dir,
                                      pileup_matrix=pileup_matrix,
                                      reference_matrix=reference_matrix,
                                      pileup_repeat_matrix=pileup_repeat_matrix,
                                      reference_repeat_matrix=reference_repeat_matrix,
                                      reversal_matrix=reversal_matrix,
                                      chromosome_name=chromosome_name,
                                      start=pileup_start)

    if counter is not None:
        counter.value += 1

        sys.stdout.write('\r' + "%.2f%% Completed" % (100 * counter.value / n_chunks))


def encode_region(bam_file_path, reference_file_path, chromosome_name, region, window_size, output_dir, runlength=False):
    length = region[1] - region[0]
    windows = chunk_interval(interval=region, chunk_size=window_size, length=length)

    for window in tqdm(windows):
        if runlength:
            generate_window_run_length_encoding(bam_file_path=bam_file_path,
                                                reference_file_path=reference_file_path,
                                                chromosome_name=chromosome_name,
                                                window=window,
                                                output_dir=output_dir)
        else:
            generate_window_encoding(bam_file_path=bam_file_path,
                                     reference_file_path=reference_file_path,
                                     chromosome_name=chromosome_name,
                                     window=window,
                                     output_dir=output_dir)


def encode_region_parallel(bam_file_path, reference_file_path, chromosome_name, region, window_size, output_dir, reverse_sort, sort_sequences_by_length, two_pass, max_threads=1, runlength=False, windows_path=None):
    length = region[1] - region[0]

    if windows_path is None:
        print("Chunking intervals")
        windows = chunk_interval(interval=region, chunk_size=window_size, length=length)
    else:
        print("Loading windows")
        windows = load_windows(windows_path)

    save_data = True
    print_results = False
    plot_results = False
    n_chunks = len(windows)

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    args_per_thread = list()
    for window in windows:
        if window[0] > window[-1]:
            print("WARNING: window excluded because invalid: ", window)
            continue

        args = [bam_file_path, reference_file_path, chromosome_name, window, output_dir, sort_sequences_by_length, reverse_sort, two_pass, save_data, print_results, plot_results, counter, n_chunks]
        args_per_thread.append(args)

    if len(args_per_thread) < max_threads:
        max_threads = len(args_per_thread)

    if runlength:
        process = generate_window_run_length_encoding
    else:
        process = generate_window_encoding

    # initiate threading
    with Pool(processes=max_threads) as pool:
        pool.starmap(process, args_per_thread)


def compare_runlength_region(bam_file_path, reference_file_path, chromosome_name, region, window_size, output_dir):
    length = region[1] - region[0]
    windows = chunk_interval(interval=region, chunk_size=window_size, length=length)

    for window in tqdm(windows):
        generate_window_encoding(bam_file_path=bam_file_path,
                                 reference_file_path=reference_file_path,
                                 chromosome_name=chromosome_name,
                                 window=window,
                                 output_dir=output_dir,
                                 save_data=False,
                                 plot_results=True)

        generate_window_run_length_encoding(bam_file_path=bam_file_path,
                                            reference_file_path=reference_file_path,
                                            chromosome_name=chromosome_name,
                                            window=window,
                                            output_dir=output_dir,
                                            save_data=False,
                                            plot_results=True)


def run_parameter_comparison():
    # ---- Nanopore GUPPY - C ELEGANS - (dev machine) -------------------------
    bam_file_path = "/home/ryan/data/Nanopore/celegans/all_chips_20k_Boreal_minimap2.sorted.bam"
    reference_file_path = "/home/ryan/data/Nanopore/celegans/GCF_000002985.6_WBcel235_genomic.fasta"
    # -------------------------------------------------------------------------

    fasta_handler = FastaHandler(reference_file_path)

    chromosomal_window_path = "/home/ryan/code/nanopore_assembly/output/window_selection/NC_003279.8_0_15072434_2018_10_12_10_58_56_199382"
    chromosome_name = "NC_003279.8"

    chromosome_length = fasta_handler.get_chr_sequence_length(chromosome_name)
    region = [0,chromosome_length]

    runlength = True

    output_root_dir = "output/"
    instance_dir = "spoa_pileup_generation_anchored_" + get_current_timestamp()
    output_dir = os.path.join(output_root_dir, instance_dir)

    encode_region_parallel(bam_file_path=bam_file_path,
                           reference_file_path=reference_file_path,
                           chromosome_name=chromosome_name,
                           region=region,
                           window_size=20,
                           output_dir=output_dir,
                           runlength=runlength,
                           max_threads=30,
                           windows_path=chromosomal_window_path,
                           sort_sequences_by_length=False,
                           reverse_sort=False,
                           two_pass=True)

    # output_root_dir = "output/"
    # instance_dir = "spoa_pileup_generation_fixed_size_" + get_current_timestamp()
    # output_dir = os.path.join(output_root_dir, instance_dir)
    #
    # encode_region_parallel(bam_file_path=bam_file_path,
    #                        reference_file_path=reference_file_path,
    #                        chromosome_name=chromosome_name,
    #                        region=region,
    #                        window_size=20,
    #                        output_dir=output_dir,
    #                        runlength=runlength,
    #                        max_threads=30,
    #                        windows_path=None,
    #                        sort_sequences_by_length=False,
    #                        reverse_sort=False,
    #                        two_pass=True)


def generate_random_intervals(n_windows, window_size, chromosome_length):
    windows = list()

    for i in range(n_windows):
        start = random.randint(0, (chromosome_length-window_size))

        stop = start + window_size - 1

        window = (start, stop)
        windows.append(window)

    return windows


def genomic_run():
    output_root_dir = "output/"
    instance_dir = "spoa_pileup_generation_" + get_current_timestamp()
    output_dir = os.path.join(output_root_dir, instance_dir)

    # ---- Nanopore GUPPY - C ELEGANS - (dev machine) -------------------------
    # bam_file_path = "/home/ryan/data/Nanopore/celegans/all_chips_20k_Boreal_minimap2.sorted.bam"
    # reference_file_path = "/home/ryan/data/Nanopore/celegans/GCF_000002985.6_WBcel235_genomic.fasta"
    # windows_path = "/home/ryan/code/nanopore_assembly/output/window_selection/NC_003283.11_0_20924180_2018_9_28_10_56"
    #
    # chromosomal_window_paths = ["/home/ryan/code/nanopore_assembly/output/window_selection/NC_003284.9_1000000_16718942_2018_10_15_0_38_27_259250",
    #                             "/home/ryan/code/nanopore_assembly/output/window_selection/NC_003283.11_1000000_19924180_2018_10_14_22_34_30_383603",
    #                             "/home/ryan/code/nanopore_assembly/output/window_selection/NC_003282.8_1000000_16493829_2018_10_14_20_52_58_386000",
    #                             "/home/ryan/code/nanopore_assembly/output/window_selection/NC_003281.10_1000000_12783801_2018_10_14_19_37_55_657683",
    #                             "/home/ryan/code/nanopore_assembly/output/window_selection/NC_003280.10_1000000_14279421_2018_10_14_18_14_1_786141",
    #                             "/home/ryan/code/nanopore_assembly/output/window_selection/NC_003279.8_1000000_14072434_2018_10_14_16_50_51_752205"]
    #
    # prefix_tokens = 2

    # ---- Nanopore GUPPY - E. Coli - (dev machine) -------------------------
    bam_file_path = "/home/ryan/data/Nanopore/ecoli/miten/r9_ecoli_reads_vs_ref.bam"
    reference_file_path = "/home/ryan/data/Nanopore/ecoli/miten/refEcoli.fasta"
    windows_path = "/home/ryan/code/nanopore_assembly/output/window_selection/gi_1000000_3641652_2018_11_12_13_50_13_296242"

    chromosomal_window_paths = ["/home/ryan/code/nanopore_assembly/output/window_selection/gi_1000000_3641652_2018_11_12_13_50_13_296242"]

    prefix_tokens = 1

    # -------------------------------------------------------------------------

    for path in reversed(chromosomal_window_paths):
        chromosome_name = "_".join(path.split("/")[-1].split("_")[0:prefix_tokens])
        print("STARTING", chromosome_name)

        region = [-1,-1]
        runlength = True

        print("USING RUNLENGTH: ", runlength)

        encode_region_parallel(bam_file_path=bam_file_path,
                               reference_file_path=reference_file_path,
                               chromosome_name=chromosome_name,
                               region=region,
                               window_size=20,
                               output_dir=output_dir,
                               runlength=runlength,
                               max_threads=30,
                               windows_path=path,
                               sort_sequences_by_length=False,
                               reverse_sort=False,
                               two_pass=True)

    print()


def main():
    output_root_dir = "output/"
    instance_dir = "spoa_pileup_generation_" + get_current_timestamp()
    output_dir = os.path.join(output_root_dir, instance_dir)

    # ---- Illumina (laptop) --------------------------------------------------
    # bam_file_path = "/Users/saureous/data/Platinum/chr1.sorted.bam"
    # reference_file_path = "/Users/saureous/data/Platinum/chr1.fa"
    # vcf_path = "/Users/saureous/data/Platinum/NA12878_S1.genome.vcf.gz"
    # bed_path = "/Users/saureous/data/Platinum/chr1_confident.bed"

    # ---- GIAB (dev machine) -------------------------------------------------
    # bam_file_path = "/home/ryan/data/GIAB/NA12878_GIAB_30x_GRCh37.sorted.bam"
    # reference_file_path = "/home/ryan/data/GIAB/GRCh37_WG.fa"
    # vcf_path = "/home/ryan/data/GIAB/NA12878_GRCh37.vcf.gz"
    # bed_path = "/home/ryan/data/GIAB/NA12878_GRCh38_confident.bed"

    # ---- Nanopore - GUPPY HUMAN - (dev machine) -----------------------------
    # bam_file_path = "/home/ryan/data/Nanopore/Human/BAM/Guppy/rel5-guppy-0.3.0-chunk10k.sorted.bam"
    # reference_file_path = "/home/ryan/data/GIAB/GRCh38_WG.fa"
    # vcf_path = "/home/ryan/data/GIAB/NA12878_GRCh38_PG.vcf.gz"
    # bed_path = "/home/ryan/data/GIAB/NA12878_GRCh38_confident.bed"

    # ---- Nanopore GUPPY - C ELEGANS - (dev machine) -------------------------
    # bam_file_path = "/home/ryan/data/Nanopore/celegans/all_chips_20k_Boreal_minimap2.sorted.bam"
    # reference_file_path = "/home/ryan/data/Nanopore/celegans/GCF_000002985.6_WBcel235_genomic.fasta"
    # windows_path = "/home/ryan/code/nanopore_assembly/output/window_selection/NC_003283.11_0_20924180_2018_9_28_10_56"

    # ---- Nanopore GUPPY - E. Coli - (dev machine) -------------------------
    # bam_file_path = "/home/ryan/data/Nanopore/ecoli/miten/r9_ecoli.contigs.fasta.reads.sorted.bam"
    bam_file_path = "/home/ryan/data/Nanopore/ecoli/miten/r9_ecoli_reads_vs_ref.bam"
    reference_file_path = "/home/ryan/data/Nanopore/ecoli/miten/refEcoli.fasta"
    # reference_file_path = "/home/ryan/data/Nanopore/ecoli/miten/r9_ecoli.contigs.fasta"

    # -------------------------------------------------------------------------

    fasta_handler = FastaHandler(reference_file_path)
    contig_names = fasta_handler.get_contig_names()


    chromosome_name = "gi"     # E coli
    # chromosome_name = "NC_003279.8"     # celegans chr1
    # chromosome_name = "NC_003283.11"     # celegans chr5
    # chromosome_name = "1"
    # chromosome_name = "chr" + chromosome_name

    lengths = list()
    for name in contig_names:
        chromosome_length = fasta_handler.get_chr_sequence_length(name)
        lengths.append(chromosome_length)

    print('\t'.join(contig_names))
    print('\t\t'.join(map(str,lengths)))

    chromosome_length = fasta_handler.get_chr_sequence_length(chromosome_name)

    # ---- TEST window --------------------------------------------------------

    # window = [762580, 762600]       # nanopore broken alignment region...  POAPY ONLY
    # window = [748460, 748480]       # nanopore broken alignment region...  POAPY ONLY
    # window = [767240, 767260]       # nanopore broken alignment region...  SPOA NOOOOooOOoooo
    # window = [727360, 767280]       # nanopore broken alignment region...  very high loss in CNNRNN
    # window = [727200, 727220]       # nanopore broken alignment region...  very high loss in CNNRNN
    # window = [748220, 748240]       # nanopore broken alignment region...  very high loss in CNNRNN
    # window = [1105084, 1105104]   # very messy alignment even with spoa... why?
    # window = [246567, 246587]     # previously failing test case for collapsed reads
    # window = [800000, 800020]

    ## test sites for misalignment
    # windows = [[10029532, 10029532+83],
    #            [10031827, 10031827+34],
    #            [10039004, 10039004+25],
    #            [10040234, 10040234+61],
    #            [1004298, 1004298+109],
    #            [10044514, 10044514+54],
    #            [10037167, 10037167+82],
    #            [14952118-5, 14952118+20]]
    #
    # use_runlength = False
    #
    # for window in windows:
    #     if use_runlength:
    #         generate_window_run_length_encoding(bam_file_path=bam_file_path,
    #                                             reference_file_path=reference_file_path,
    #                                             chromosome_name=chromosome_name,
    #                                             window=window,
    #                                             output_dir=output_dir,
    #                                             sort_sequences_by_length=False,
    #                                             reverse_sort=False,
    #                                             two_pass=True,
    #                                             plot_results=False,
    #                                             print_results=True,
    #                                             save_data=False,
    #                                             save_to_fasta=True)
    #     else:
    #         generate_window_encoding(bam_file_path=bam_file_path,
    #                                  reference_file_path=reference_file_path,
    #                                  chromosome_name=chromosome_name,
    #                                  window=window,
    #                                  output_dir=output_dir,
    #                                  sort_sequences_by_length=False,
    #                                  reverse_sort=False,
    #                                  two_pass=True,
    #                                  plot_results=True,
    #                                  print_results=True,
    #                                  save_data=False,
    #                                  save_to_fasta=True)

    # ---- generate random windows ---------------------------------------------

    # windows = set(map(tuple,load_windows("/home/ryan/code/nanopore_assembly/output/window_selection/NC_003279.8_1000000_14072434_2018_10_14_16_50_51_752205")))
    # n_windows = 2000
    #
    # for i in range(n_windows):
    #     sys.stdout.write("\r %.3f" % 100*(float(i)/n_windows))
    #     window = windows.pop()
    #
    #     # print("GENERATING WINDOW: ", window)
    #
    #     generate_window_fasta(bam_file_path=bam_file_path,
    #                           reference_file_path=reference_file_path,
    #                           chromosome_name=chromosome_name,
    #                           window=window,
    #                           output_dir=output_dir)


    # ---- generate random windows ---------------------------------------------

    # n_windows = 20
    # window_size = 10000
    #
    # i = 0
    # while i < n_windows:
    #     sys.stdout.write("\r %.3f" % (100*(float(i)/n_windows)))
    #
    #     start = random.randint(0, (chromosome_length-window_size))
    #     stop = start + window_size - 1
    #
    #     window = (start, stop)
    #
    #     print("\nGENERATING WINDOW: ", window)
    #
    #     reads_found = generate_window_fasta(bam_file_path=bam_file_path,
    #                                         reference_file_path=reference_file_path,
    #                                         chromosome_name=chromosome_name,
    #                                         window=window,
    #                                         output_dir=output_dir,
    #                                         exclude_loose_ends=False)
    #
    #     if reads_found:
    #         i += 1


    # ---- TEST region --------------------------------------------------------

    # region = [2000000, 12000000]
    # region = [0, chromosome_length]
    # region = [-1,-1]
    # runlength = True
    #
    # encode_region_parallel(bam_file_path=bam_file_path,
    #                        reference_file_path=reference_file_path,
    #                        chromosome_name=chromosome_name,
    #                        region=region,
    #                        window_size=20,
    #                        output_dir=output_dir,
    #                        runlength=runlength,
    #                        max_threads=30,
    #                        windows_path=windows_path)

    # ---- COMPARE runlength vs standard encoding for region ------------------

    # region = [800000, 850000]  # illumina laptop test region
    #
    # compare_runlength_region(bam_file_path=bam_file_path,
    #                          reference_file_path=reference_file_path,
    #                          chromosome_name=chromosome_name,
    #                          region=region,
    #                          window_size=20,
    #                          output_dir=output_dir)

    # ---- genomic run --------------------------------------------------------

    # runlength = True
    #
    # start_position = 0
    # end_position = 250000000
    # #
    # generate_data(bam_file_path=bam_file_path,
    #               reference_file_path=reference_file_path,
    #               vcf_path=vcf_path,
    #               bed_path=bed_path,
    #               chromosome_name=chromosome_name,
    #               start_position=start_position,
    #               end_position=end_position,
    #               output_dir=output_dir,
    #               runlength=runlength)


if __name__ == "__main__":
    # main()
    genomic_run()
    # run_parameter_comparison()