from modules.PileupGenerator import SegmentGrabber
from modules.IntervalTree import IntervalTree
from handlers.VcfHandler import VCFFileProcessor
from handlers.FastaHandler import FastaHandler
from handlers.BamHandler import BamHandler
from handlers.TsvHandler import TsvHandler
from select_windows import WINDOW_SIZE, CDF_STEP_SIZE
from modules.window_selection_utils import merge_windows
from modules.pileup_utils import *
from modules.alignment_utils import *
from multiprocessing import Pool
import multiprocessing
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


def get_aligned_segments(fasta_handler, bam_handler, chromosome_name, pileup_start, pileup_end, include_ref=False):
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

    segment_grabber = SegmentGrabber(chromosome_name=chromosome_name,
                                     start_position=pileup_start,
                                     end_position=pileup_end,
                                     ref_sequence=ref_sequence,
                                     reads=reads)

    # if a reference sequence is intended to be added to the pileup, then leave a space for it
    if include_ref:
        segment_grabber.max_coverage -= 1

    sequence_dictionary = segment_grabber.get_read_segments()

    # for key in sequence_dictionary:
    #     print(sequence_dictionary[key])

    exit()

    # if len(sequence_dictionary.keys()) == 0:
    #     print("\nWARNING: No reads found at position:", pileup_start, pileup_end)
    #     ref_sequence = None
    #     read_ids = None
    #     sequences = None
    #     reversal_statuses = None
    # else:
    #     read_ids = list()
    #     sequences = list()
    #     reversal_statuses = list()
    #
    #     for read_id in sequence_dictionary:
    #         sequence = sequence_dictionary[read_id]
    #         reverse_status = reverse_status_dictionary[read_id]
    #
    #         sequences.append(sequence)
    #         reversal_statuses.append(reverse_status)

    # return ref_sequence, read_ids, sequences, reversal_statuses


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


def generate_window_run_length_encoding(bam_file_path, reference_file_path, chromosome_name, window, output_dir, sort_sequences_by_length=False, reverse_sort=False, two_pass=False, save_data=True, print_results=False, plot_results=False, counter=None, n_chunks=None):
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
        x_repeat = pileup_repeat_matrix.reshape([1, height, width])
        y_repeat = reference_repeat_matrix.reshape([1, width])
        reversal = reversal_matrix.reshape([1, height, width])

        x_pileup_flat = flatten_one_hot_tensor(x_pileup)
        y_pileup_flat = flatten_one_hot_tensor(y_pileup)
        plot_runlength_prediction_stranded(x_pileup=x_pileup_flat,
                                           x_repeat=x_repeat.squeeze(),
                                           y_pileup=y_pileup_flat,
                                           y_repeat=y_repeat,
                                           reversal=reversal.squeeze(),
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


def encode_region_parallel(bam_file_path, reference_file_path, chromosome_name, region, window_size, output_dir, max_threads=1, runlength=False, windows_path=None):
    length = region[1] - region[0]

    if windows_path is None:
        windows = chunk_interval(interval=region, chunk_size=window_size, length=length)
    else:
        windows = load_windows(windows_path)

    save_data = True
    print_results = False
    plot_results = False
    n_chunks = len(windows)

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    args_per_thread = list()
    for window in windows:
        args = [bam_file_path, reference_file_path, chromosome_name, window, output_dir, save_data, print_results, plot_results, counter, n_chunks]
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


def genomic_run():
    output_root_dir = "output/"
    instance_dir = "spoa_pileup_generation_" + get_current_timestamp()
    output_dir = os.path.join(output_root_dir, instance_dir)

    # ---- Nanopore GUPPY - C ELEGANS - (dev machine) -------------------------
    bam_file_path = "/home/ryan/data/Nanopore/celegans/all_chips_20k_Boreal_minimap2.sorted.bam"
    reference_file_path = "/home/ryan/data/Nanopore/celegans/GCF_000002985.6_WBcel235_genomic.fasta"
    windows_path = "/home/ryan/code/nanopore_assembly/output/window_selection/NC_003283.11_0_20924180_2018_9_28_10_56"
    # -------------------------------------------------------------------------

    fasta_handler = FastaHandler(reference_file_path)

    chromosomal_window_paths = ["output/window_selection/NC_003279.8_0_15072434_2018_10_1_20_1",
                                "output/window_selection/NC_003280.10_0_15279421_2018_10_1_21_25",
                                "output/window_selection/NC_003281.10_0_13783801_2018_10_1_22_40",
                                "output/window_selection/NC_003282.8_0_17493829_2018_10_1_23_51",
                                "output/window_selection/NC_003283.11_0_20924180_2018_10_2_1_22",
                                "output/window_selection/NC_003284.9_0_17718942_2018_10_2_3_10",
                                "output/window_selection/NC_001328.1_0_13794_2018_10_2_4_46"]

    for path in chromosomal_window_paths:
        chromosome_name = "_".join(path.split("/")[-1].split("_")[0:2])
        print("STARTING", chromosome_name)

        region = [-1,-1]
        runlength = True

        encode_region_parallel(bam_file_path=bam_file_path,
                               reference_file_path=reference_file_path,
                               chromosome_name=chromosome_name,
                               region=region,
                               window_size=20,
                               output_dir=output_dir,
                               runlength=runlength,
                               max_threads=30,
                               windows_path=path)



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
    bam_file_path = "/home/ryan/data/Nanopore/celegans/all_chips_20k_Boreal_minimap2.sorted.bam"
    reference_file_path = "/home/ryan/data/Nanopore/celegans/GCF_000002985.6_WBcel235_genomic.fasta"
    windows_path = "/home/ryan/code/nanopore_assembly/output/window_selection/NC_003283.11_0_20924180_2018_9_28_10_56"
    # -------------------------------------------------------------------------

    fasta_handler = FastaHandler(reference_file_path)
    contig_names = fasta_handler.get_contig_names()

    chromosome_name = "NC_003279.8"     # celegans chr1
    # chromosome_name = "NC_003283.11"     # celegans chr5
    # chromosome_name = "1"
    # chromosome_name = "chr" + chromosome_name

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

    # test sites for misalignment
    # window = [10029532, 10029532+83]
    # window = [10031827, 10031827+34]
    # window = [10039004, 10039004+25]
    # window = [10040234, 10040234+61]
    # window = [1004298, 1004298+109]
    window = [10044514, 10044514+54]
    # window = [10037167, 10037167+82]

    # test_window(bam_file_path=bam_file_path,
    #             reference_file_path=reference_file_path,
    #             chromosome_name=chromosome_name,
    #             window=window,
    #             output_dir=output_dir,
    #             print_results=True,
    #             save_data=True)

    generate_window_run_length_encoding(bam_file_path=bam_file_path,
                                        reference_file_path=reference_file_path,
                                        chromosome_name=chromosome_name,
                                        window=window,
                                        output_dir=output_dir,
                                        sort_sequences_by_length=True,
                                        reverse_sort=False,
                                        two_pass=True,
                                        plot_results=True,
                                        print_results=True,
                                        save_data=False)

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
    main()
    # genomic_run()