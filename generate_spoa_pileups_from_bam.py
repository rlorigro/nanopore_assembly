from modules.PileupGenerator import PileupGenerator
from modules.IntervalTree import IntervalTree
from handlers.VcfHandler import VCFFileProcessor
from handlers.FastaHandler import FastaHandler
from handlers.BamHandler import BamHandler
from handlers.TsvHandler import TsvHandler
from modules.pileup_utils import *
from modules.alignment_utils import *
from matplotlib import pyplot
from datetime import datetime
from tqdm import tqdm
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


def collapse_repeats(sequences):
    """
    For a list of sequences, collapse repeated characters to single characters, and generate a list of integer values
    that describe the number of repeats for each character
    :param sequences:
    :return:
    """
    character_sequences = list()
    character_counts = list()

    for sequence in sequences:
        character_sequence = list()
        character_count = list()
        current_character = None

        for character in sequence:
            if character != current_character:
                character_sequence.append(character)
                character_count.append(1)
            else:
                character_count[-1] += 1

            current_character = character

        character_sequence = ''.join(character_sequence)

        character_sequences.append(character_sequence)
        character_counts.append(character_count)

    return character_sequences, character_counts


def get_aligned_segments(fasta_handler, bam_handler, chromosome_name, pileup_start, pileup_end):
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

    pileup_generator = PileupGenerator(chromosome_name=chromosome_name,
                                       start_position=pileup_start,
                                       end_position=pileup_end,
                                       ref_sequence=ref_sequence,
                                       reads=reads)

    sequence_dictionary = pileup_generator.get_read_segments()

    if len(sequence_dictionary.keys()) == 0:
        exit("No reads found at position")

    read_ids, sequences = zip(*sequence_dictionary.items())

    return ref_sequence, read_ids, sequences


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


def generate_data(bam_file_path, reference_file_path, vcf_path, bed_path, chromosome_name, start_position, end_position, output_dir, generate_from_vcf=False, runlength=False):
    """
    Generate pileups from BAM data, and collapse sequences to have no explicitly repeated characters. Additionally
    encode a repeat channel that describes the number of repeats observed per base.
    :param bam_file_path:
    :param reference_file_path:
    :param vcf_path:
    :param chromosome_name:
    :param start_position:
    :param end_position:
    :return:
    """
    if runlength:
        encode_window = generate_window_run_length_encoding
    else:
        encode_window = generate_window_encoding

    if generate_from_vcf:
        chromosomal_windows = get_variant_windows(vcf_path=vcf_path,
                                                  chromosome_name=chromosome_name,
                                                  start_position=start_position,
                                                  end_position=end_position)

    else:
        chromosomal_windows = get_non_variant_windows(vcf_path=vcf_path,
                                                      bed_path=bed_path,
                                                      chromosome_name=chromosome_name,
                                                      start_position=start_position,
                                                      end_position=end_position)

    for chromosome_name in chromosomal_windows:
        for w,window in enumerate(chromosomal_windows[chromosome_name]):
            pileup_start = window[0]
            pileup_end = window[1]      # add random variation here

            print(pileup_start, pileup_end)

            encode_window(bam_file_path=bam_file_path,
                          reference_file_path=reference_file_path,
                          chromosome_name=chromosome_name,
                          window=window,
                          output_dir=output_dir,
                          save_data=True,
                          print_results=False,
                          plot_results=False)


def generate_window_encoding(bam_file_path, reference_file_path, chromosome_name, window, output_dir, save_data=True, print_results=False, plot_results=False):
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

    ref_sequence, read_ids, sequences = get_aligned_segments(fasta_handler=fasta_handler,
                                                             bam_handler=bam_handler,
                                                             chromosome_name=chromosome_name,
                                                             pileup_start=pileup_start,
                                                             pileup_end=pileup_end)

    alignments, ref_alignment = get_spoa_alignment(sequences=sequences, ref_sequence=ref_sequence)
    ref_alignment = [ref_alignment]

    pileup_matrix = convert_alignments_to_matrix(alignments)
    reference_matrix = convert_aligned_reference_to_one_hot(ref_alignment)

    if plot_results:
        plot_encodings(pileup_matrix=pileup_matrix, reference_matrix=reference_matrix)

    if print_results:
        print_segments(ref_sequence, sequences)

        for label, alignstring in alignments:
            print("{0:15s} {1:s}".format(label, alignstring))

        for label, alignstring in ref_alignment:
            print("{0:15s} {1:s}".format(label, alignstring))

        # visualize_matrix(pileup_matrix)
        # visualize_matrix(reference_matrix)

    if ref_alignment[0][1].replace("-",'') != ref_sequence:
        print("Aligned reference does not match true reference at [%d,%d]"%(pileup_start,pileup_end))
        print("unaligned:\t",ref_sequence)
        print("aligned:\t",ref_alignment[0][1].replace("-",''))

    elif save_data:
        save_training_data(output_dir=output_dir,
                           pileup_matrix=pileup_matrix,
                           reference_matrix=reference_matrix,
                           chromosome_name=chromosome_name,
                           start=pileup_start)


def generate_window_run_length_encoding(bam_file_path, reference_file_path, chromosome_name, window, output_dir, save_data=True, print_results=False, plot_results=False):
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

    ref_sequence, read_ids, sequences = get_aligned_segments(fasta_handler=fasta_handler,
                                                             bam_handler=bam_handler,
                                                             chromosome_name=chromosome_name,
                                                             pileup_start=pileup_start,
                                                             pileup_end=pileup_end)

    sequences, repeats = collapse_repeats(sequences)
    ref_sequence, ref_repeats = collapse_repeats([ref_sequence])

    ref_sequence = ref_sequence[0]

    alignments, ref_alignment = get_spoa_alignment(sequences=sequences, ref_sequence=ref_sequence)
    ref_alignment = [ref_alignment]

    pileup_matrix, pileup_repeat_matrix = convert_collapsed_alignments_to_matrix(alignments, repeats)
    reference_matrix, reference_repeat_matrix = convert_collapsed_alignments_to_matrix(ref_alignment, ref_repeats, fixed_coverage=False)

    if plot_results:
        plot_collapsed_encodings(pileup_matrix=pileup_matrix,
                                 reference_matrix=reference_matrix,
                                 pileup_repeat_matrix=pileup_repeat_matrix,
                                 reference_repeat_matrix=reference_repeat_matrix)

    if print_results:
        print_segments(ref_sequence, sequences)

        for label, alignstring in alignments:
            print("{0:15s} {1:s}".format(label, alignstring))

        for label, alignstring in ref_alignment:
            print("{0:15s} {1:s}".format(label, alignstring))

        print(repeats)
        print(ref_repeats)

    if ref_alignment[0][1].replace("-",'') != ref_sequence:
        print("Aligned reference does not match true reference at [%d,%d]"%(pileup_start,pileup_end))
        print("unaligned:\t",ref_sequence)
        print("aligned:\t",ref_alignment[0][1].replace("-",''))

    elif save_data:
        save_run_length_training_data(output_dir=output_dir,
                                      pileup_matrix=pileup_matrix,
                                      reference_matrix=reference_matrix,
                                      pileup_repeat_matrix=pileup_repeat_matrix,
                                      reference_repeat_matrix=reference_repeat_matrix,
                                      chromosome_name=chromosome_name,
                                      start=pileup_start)


def test_region(bam_file_path, reference_file_path, chromosome_name, region, window_size, output_dir, runlength=False):
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

    # ---- Nanopore GUPPY (dev machine) --------------------------------------
    bam_file_path = "/home/ryan/data/Nanopore/Human/BAM/Guppy/rel5-guppy-0.3.0-chunk10k.sorted.bam"
    reference_file_path = "/home/ryan/data/GIAB/GRCh38_WG.fa"
    vcf_path = "/home/ryan/data/GIAB/NA12878_GRCh38_PG.vcf.gz"
    bed_path = "/home/ryan/data/GIAB/NA12878_GRCh38_confident.bed"
    # -------------------------------------------------------------------------

    chromosome_name = "1"
    chromosome_name = "chr" + chromosome_name

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

    # FIX THIS BUG:
    # [0.6 0.4 0.6 0.4]
    # ERROR: incorrect dimensions for pileup:
    #     / home / ryan / code / nanopore_assembly / output / spoa_pileup_generation_2018 - 9 - 6 - 13 - 16 - 52 - 3 - 249 / chr1 / chr1_1007747_matrix.npz

    # test_window(bam_file_path=bam_file_path,
    #             reference_file_path=reference_file_path,
    #             chromosome_name=chromosome_name,
    #             window=window,
    #             output_dir=output_dir,
    #             print_results=True,
    #             save_data=True)

    # test_window_run_length_encoding(bam_file_path=bam_file_path,
    #                                 reference_file_path=reference_file_path,
    #                                 chromosome_name=chromosome_name,
    #                                 window=window,
    #                                 output_dir=output_dir,
    #                                 print_results=True,
    #                                 save_data=True)

    # ---- TEST region --------------------------------------------------------

    # region = [800000, 850000]
    # runlength = True
    #
    # test_region(bam_file_path=bam_file_path,
    #             reference_file_path=reference_file_path,
    #             chromosome_name=chromosome_name,
    #             region=region,
    #             window_size=20,
    #             output_dir=output_dir,
    #             runlength=runlength)

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

    runlength = False

    start_position = 0
    end_position = 250000000
    #
    generate_data(bam_file_path=bam_file_path,
                  reference_file_path=reference_file_path,
                  vcf_path=vcf_path,
                  bed_path=bed_path,
                  chromosome_name=chromosome_name,
                  start_position=start_position,
                  end_position=end_position,
                  output_dir=output_dir,
                  runlength=runlength)


if __name__ == "__main__":
    main()
