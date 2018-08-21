from modules.PileupGenerator import PileupGenerator
from modules.IntervalTree import IntervalTree
from handlers.VcfHandler import VCFFileProcessor
from handlers.FastaHandler import FastaHandler
from handlers.BamHandler import BamHandler
from handlers.TsvHandler import TsvHandler
from poapy import seqgraphalignment
from poapy import poagraph
from matplotlib import pyplot
import numpy
import math


sequence_to_float = {"-":0,
                     "A":1,
                     "G":2,
                     "T":3,
                     "C":4}

# How to encode bases in a single 0-1 channel
SCALE_FACTOR = 0.2

# POA alignment parameters
FAST = True
GLOBAL_ALIGN = True
MATCH_SCORE = 4
MISMATCH_SCORE = -4
GAP_SCORE = -5


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
        index = sequence_to_float[character]

        matrix[index,c] = 1

    # print(matrix)

    pyplot.imshow(matrix, cmap="viridis")
    pyplot.show()


def convert_reference_to_one_hot(reference_sequence):
    """
    given a reference sequence, generate an lx4 matrix of one-hot encodings where l=sequence length and 4 is the # of nt
    :param reference_sequence:
    :return:
    """
    length = len(reference_sequence)
    matrix = numpy.zeros([4,length])

    for c,character in enumerate(reference_sequence):
        index = sequence_to_float[character] - 1

        matrix[index,c] = 1

    # print(matrix)

    pyplot.imshow(matrix, cmap="viridis")
    pyplot.show()


def convert_collapsed_alignments_to_matrix(alignments, character_counts):
    """
    For a list of alignment strings, generate a matrix of encoded bases in float format from 0-1
    :param alignments:
    :return:
    """
    n = len(alignments)
    m = len(alignments[0][1])

    base_matrix = numpy.zeros([n, m])
    repeat_matrix = numpy.zeros([n,m])

    for a,alignment in enumerate(alignments):
        c = 0
        read_id, alignment_string = alignment

        for b,character in enumerate(alignment_string):
            base_matrix[a,b] = sequence_to_float[character]*SCALE_FACTOR

            if character != "-":
                # print("a", a, "c", c, "len c", len(character_counts[a]), alignment[1], character_counts[a][c], character_counts[a])

                # print(a, c)
                # print(alignment_string.replace("-",''))
                # print(''.join(map(str,character_counts[a])))

                repeat_matrix[a,b] = character_counts[a][c]
                c += 1

    pyplot.imshow(base_matrix, cmap="viridis")
    pyplot.show()
    pyplot.imshow(repeat_matrix, cmap="viridis")
    pyplot.show()

    return base_matrix, repeat_matrix


def convert_alignments_to_matrix(alignments):
    """
    For a list of alignment strings, generate a matrix of encoded bases in float format from 0-1
    :param alignments:
    :return:
    """
    n = len(alignments)
    m = len(alignments[0][1])

    matrix = numpy.zeros([n, m])
    for a,alignment in enumerate(alignments):
        read_id, alignment_string = alignment

        for b,character in enumerate(alignment_string):
            matrix[a,b] = sequence_to_float[character]*SCALE_FACTOR

    # print(matrix)

    pyplot.imshow(matrix, cmap="viridis")
    pyplot.show()

    return matrix


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

        # print(sequence)
        # print(''.join(character_sequence))
        # print(''.join(map(str,character_count)))

    return character_sequences, character_counts


def partial_order_alignment(ref_sequence, sequences, graph=None, include_reference=False):
    """
    Generate partial order alignment graph and find fixed global read alignment string for each read
    :param ref_sequence:
    :param sequences:
    :return:
    """

    if graph is None:
        init_sequence = sequences[0]
        init_label = "0"

        graph = poagraph.POAGraph(init_sequence, init_label)

    for i in range(1, len(sequences)):
        sequence = sequences[i]

        alignment = seqgraphalignment.SeqGraphAlignment(sequence, graph,
                                                        fastMethod=FAST,
                                                        globalAlign=GLOBAL_ALIGN,
                                                        matchscore=MATCH_SCORE,
                                                        mismatchscore=MISMATCH_SCORE,
                                                        gapscore=GAP_SCORE)

        graph.incorporateSeqAlignment(alignment, sequence, str(i))

    if include_reference:
        alignment = seqgraphalignment.SeqGraphAlignment(ref_sequence, graph,
                                                        fastMethod=FAST,
                                                        globalAlign=GLOBAL_ALIGN,
                                                        matchscore=MATCH_SCORE,
                                                        mismatchscore=MISMATCH_SCORE,
                                                        gapscore=GAP_SCORE)

        graph.incorporateSeqAlignment(alignment, ref_sequence, "ref")

    alignments = graph.generateAlignmentStrings(consensus=False)

    return alignments, graph


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

            alignments, graph = partial_order_alignment(ref_sequence, sequences, include_reference=True)

            ref_alignment = alignments[-1:]
            alignments = alignments[:-1]

            for label, alignstring in alignments:
                print("{0:15s} {1:s}".format(label, alignstring))

            convert_alignments_to_matrix(alignments)
            # convert_alignments_to_matrix(ref_alignment)
            # convert_reference_to_one_hot(ref_sequence)
            convert_aligned_reference_to_one_hot(ref_alignment)

            assert ref_alignment[0][1].replace("-",'') == ref_sequence, "Aligned reference does not match true reference at [%d,%d]"%(pileup_start,pileup_end)

            if w == 0:
                with open("test.html", 'w') as file:
                    graph.htmlOutput(file)

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

            alignments, graph = partial_order_alignment(ref_sequence, character_sequences, include_reference=True)

            ref_alignment = alignments[-1:]
            alignments = alignments[:-1]

            for label, alignstring in alignments:
                print("{0:15s} {1:s}".format(label, alignstring))

            convert_collapsed_alignments_to_matrix(alignments, character_counts)
            # convert_collapsed_alignments_to_matrix(alignments, character_counts)
            # convert_reference_to_one_hot(ref_sequence)
            convert_aligned_reference_to_one_hot(ref_alignment)

            if w == 0:
                with open("test.html", 'w') as file:
                    graph.htmlOutput(file)

            if w == 0:
                exit()


def test_window(bam_file_path, reference_file_path, chromosome_name, window):
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
    pileup_end = window[1]      # add random variation here

    print(pileup_start, pileup_end)

    ref_sequence, read_ids, sequences = get_aligned_segments(fasta_handler=fasta_handler,
                                                             bam_handler=bam_handler,
                                                             chromosome_name=chromosome_name,
                                                             pileup_start=pileup_start,
                                                             pileup_end=pileup_end)

    print_segments(ref_sequence, sequences)

    alignments, graph = partial_order_alignment(ref_sequence, sequences, include_reference=True)

    ref_alignment = alignments[-1:]
    alignments = alignments[:-1]

    for label, alignstring in alignments:
        print("{0:15s} {1:s}".format(label, alignstring))

    for label, alignstring in ref_alignment:
        print("{0:15s} {1:s}".format(label, alignstring))

    convert_alignments_to_matrix(alignments)
    # convert_alignments_to_matrix(ref_alignment)
    # convert_reference_to_one_hot(ref_sequence)
    convert_aligned_reference_to_one_hot(ref_alignment)

    if ref_alignment[0][1].replace("-",'') != ref_sequence:
        print("Aligned reference does not match true reference at [%d,%d]"%(pileup_start,pileup_end))

    print("unaligned:\t",ref_sequence)
    print("aligned:\t",ref_alignment[0][1].replace("-",''))

    with open("test.html", 'w') as file:
        graph.htmlOutput(file)



def main():
    chromosome_name = "18"

    # ---- GIAB (dev machine) -------------------------------------------------
    # bam_file_path = "/home/ryan/data/GIAB/NA12878_GIAB_30x_GRCh37.sorted.bam"
    # reference_file_path = "/home/ryan/data/GIAB/GRCh37_WG.fa"
    # vcf_path = "/home/ryan/data/GIAB/NA12878_GRCh37.vcf.gz"

    # ---- Nanopore GUPPY (dev machine) --------------------------------------
    bam_file_path = "/home/ryan/data/Nanopore/Human/BAM/Guppy/rel5-guppy-0.3.0-chunk10k.sorted.bam"
    reference_file_path = "/home/ryan/data/GIAB/GRCh38_WG.fa"
    vcf_path = "/home/ryan/data/GIAB/NA12878_GRCh38_PG.vcf.gz"
    bed_path = "/home/ryan/data/GIAB/NA12878_GRCh38_confident.bed"

    chromosome_name = "chr" + chromosome_name
    # -------------------------------------------------------------------------

    start_position = 0
    end_position = 250000000

    test_window(bam_file_path=bam_file_path,
                reference_file_path=reference_file_path,
                chromosome_name=chromosome_name,
                window=[246567,246587])

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
