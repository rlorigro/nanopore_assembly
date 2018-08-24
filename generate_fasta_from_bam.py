from modules.PileupGenerator import PileupGenerator
from modules.IntervalTree import IntervalTree
from handlers.VcfHandler import VCFFileProcessor
from handlers.FastaHandler import FastaHandler
from handlers.FastaWriter import FastaWriter
from handlers.BamHandler import BamHandler
from handlers.TsvHandler import TsvHandler
from handlers.FileManager import FileManager
from poapy import seqgraphalignment
from poapy import poagraph
from matplotlib import pyplot
from datetime import datetime
from tqdm import tqdm
import numpy
import math
import os.path


MAX_COVERAGE = 50


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
                                       reads=reads,
                                       max_coverage=MAX_COVERAGE)

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


def generate_data(bam_file_path, reference_file_path, vcf_path, bed_path, chromosome_name, start_position, end_position, generate_from_vcf=False):
    """
    Generate pileup for read segments aligned between two genomic coordinates
    :param bam_file_path:
    :param reference_file_path:
    :param vcf_path:
    :param chromosome_name:
    :param start_position:
    :param end_position:
    :return:
    """
    bam_handler = BamHandler(bam_file_path)
    fasta_handler = FastaHandler(reference_file_path)

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

            ref_sequence, read_ids, sequences = get_aligned_segments(fasta_handler=fasta_handler,
                                                                     bam_handler=bam_handler,
                                                                     chromosome_name=chromosome_name,
                                                                     pileup_start=pileup_start,
                                                                     pileup_end=pileup_end)


            if w == 10:
                exit()


def generate_collapsed_data(bam_file_path, reference_file_path, vcf_path, bed_path, chromosome_name, start_position, end_position, generate_from_vcf=False):
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
    bam_handler = BamHandler(bam_file_path)
    fasta_handler = FastaHandler(reference_file_path)

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

            ref_sequence, read_ids, sequences = get_aligned_segments(fasta_handler=fasta_handler,
                                                                     bam_handler=bam_handler,
                                                                     chromosome_name=chromosome_name,
                                                                     pileup_start=pileup_start,
                                                                     pileup_end=pileup_end)

            character_sequences, character_counts = collapse_repeats(sequences)
            print_collapsed_segments(character_sequences, character_counts)

            if w == 0:
                exit()


def test_window(bam_file_path, reference_file_path, chromosome_name, window, output_dir, save_data=True, print_results=False):
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

    if print_results:
        print_segments(ref_sequence, sequences)

    if save_data:
        filename = "test_" + str(pileup_start)
        output_path = os.path.join(output_dir, filename)

        if not os.path.exists(output_dir):
            FileManager.ensure_directory_exists(output_dir)

        fasta_writer = FastaWriter(output_path)
        fasta_writer.write_sequences(sequences)


def test_region(bam_file_path, reference_file_path, chromosome_name, region, window_size, output_dir):
    length = region[1] - region[0]
    windows = chunk_interval(interval=region, chunk_size=window_size, length=length)

    for window in tqdm(windows):
        test_window(bam_file_path=bam_file_path,
                    reference_file_path=reference_file_path,
                    chromosome_name=chromosome_name,
                    window=window,
                    output_dir=output_dir)


def main():
    output_root_dir = "output/"
    instance_dir = "pileup_generation_" + get_current_timestamp()
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

    # window = [762580, 762600]       # nanopore broken alignment region...
    window = [748460, 748480]       # nanopore broken alignment region...
    # window = [246567, 246587]     # previously failing test case for collapsed reads

    test_window(bam_file_path=bam_file_path,
                reference_file_path=reference_file_path,
                chromosome_name=chromosome_name,
                window=window,
                output_dir=output_dir,
                print_results=True,
                save_data=True)

    # ---- TEST region --------------------------------------------------------

    # region = [715000, 785000]  # illumina laptop test region
    #
    # test_region(bam_file_path=bam_file_path,
    #             reference_file_path=reference_file_path,
    #             chromosome_name=chromosome_name,
    #             region=region,
    #             window_size=20,
    #             output_dir=output_dir)

    # ---- genomic run --------------------------------------------------------

    # start_position = 0
    # end_position = 250000000
    #
    # generate_data(bam_file_path=bam_file_path,
    #               reference_file_path=reference_file_path,
    #               vcf_path=vcf_path,
    #               bed_path=bed_path,
    #               chromosome_name=chromosome_name,
    #               start_position=start_position,
    #               end_position=end_position)

    # generate_collapsed_data(bam_file_path=bam_file_path,
    #                         reference_file_path=reference_file_path,
    #                         vcf_path=vcf_path,
    #                         bed_path=bed_path,
    #                         chromosome_name=chromosome_name,
    #                         start_position=start_position,
    #                         end_position=end_position)


if __name__ == "__main__":
    main()
