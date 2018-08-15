from handlers.BamHandler import BamHandler
from handlers.FastaHandler import FastaHandler
from handlers.VcfHandler import VCFFileProcessor
from modules.PileupGenerator import PileupGenerator
from poapy import poagraph
from poapy import seqgraphalignment
from matplotlib import pyplot
import numpy

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

                print(a, c)
                print(alignment_string.replace("-",''))
                print(''.join(map(str,character_counts[a])))

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


def partial_order_alignment(ref_sequence, sequences, include_reference=False):
    """
    Generate partial order alignment graph and find fixed global read alignment string for each read
    :param ref_sequence:
    :param sequences:
    :return:
    """
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


def generate_data(bam_file_path, reference_file_path, vcf_path, chromosome_name, start_position, end_position):
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
    vcf_handler = VCFFileProcessor(vcf_path)

    vcf_handler.populate_dictionary(contig=chromosome_name,
                                    start_pos=start_position,
                                    end_pos=end_position+1)

    positional_variants = vcf_handler.get_variant_dictionary()

    for p,position in enumerate(positional_variants):
        pileup_start = position - 5
        pileup_end = position + 5      # add random variation here

        print(pileup_start, pileup_end)

        ref_sequence, read_ids, sequences = get_aligned_segments(fasta_handler=fasta_handler,
                                                                 bam_handler=bam_handler,
                                                                 chromosome_name=chromosome_name,
                                                                 pileup_start=pileup_start,
                                                                 pileup_end=pileup_end)

        alignments, graph = partial_order_alignment(ref_sequence, sequences)

        convert_alignments_to_matrix(alignments)
        # convert_alignments_to_matrix(alignments[-1:])
        convert_reference_to_one_hot(ref_sequence)

        for label, alignstring in alignments:
            print("{0:15s} {1:s}".format(label, alignstring))

        if p == 0:
            with open("test.html", 'w') as file:
                graph.htmlOutput(file)

        if p == 0:
            exit()


def generate_collapsed_data(bam_file_path, reference_file_path, vcf_path, chromosome_name, start_position, end_position):
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
    vcf_handler = VCFFileProcessor(vcf_path)

    vcf_handler.populate_dictionary(contig=chromosome_name,
                                    start_pos=start_position,
                                    end_pos=end_position+1)

    positional_variants = vcf_handler.get_variant_dictionary()

    for p,position in enumerate(positional_variants):
        pileup_start = position - 5
        pileup_end = position + 5      # add random variation here

        print(pileup_start, pileup_end)

        ref_sequence, read_ids, sequences = get_aligned_segments(fasta_handler=fasta_handler,
                                                                 bam_handler=bam_handler,
                                                                 chromosome_name=chromosome_name,
                                                                 pileup_start=pileup_start,
                                                                 pileup_end=pileup_end)

        character_sequences, character_counts = collapse_repeats(sequences)
        print_collapsed_segments(character_sequences, character_counts)

        alignments, graph = partial_order_alignment(ref_sequence, character_sequences)

        for label, alignstring in alignments:
            print("{0:15s} {1:s}".format(label, alignstring))

        convert_collapsed_alignments_to_matrix(alignments, character_counts)
        # convert_alignments_to_matrix(alignments[-1:])
        convert_reference_to_one_hot(ref_sequence)

        if p == 0:
            with open("test.html", 'w') as file:
                graph.htmlOutput(file)

        if p == 4:
            exit()


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

    start_position = 1663352 - 10
    end_position = 1663352 + 10000000

    # generate_data(bam_file_path=bam_file_path,
    #               reference_file_path=reference_file_path,
    #               vcf_path=vcf_path,
    #               chromosome_name=chromosome_name,
    #               start_position=start_position,
    #               end_position=end_position)

    generate_collapsed_data(bam_file_path=bam_file_path,
                            reference_file_path=reference_file_path,
                            vcf_path=vcf_path,
                            chromosome_name=chromosome_name,
                            start_position=start_position,
                            end_position=end_position)

if __name__ == "__main__":
    main()