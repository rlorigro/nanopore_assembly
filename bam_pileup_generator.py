from handlers.BamHandler import BamHandler
from handlers.FastaHandler import FastaHandler
from handlers.VcfHandler import VCFFileProcessor
from modules.PileupGenerator import PileupGenerator
from poapy import poagraph
from poapy import seqgraphalignment


def print_segments(ref_sequence, sequences):
    print(ref_sequence)
    print("-" * len(ref_sequence))
    for sequence in sequences:
        print(sequence)
    print()


if __name__ == "__main__":
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

    start_position = 1000000
    end_position = 19000000

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

        print_segments(ref_sequence, sequences)

        init_sequence = sequences[0]
        init_label = read_ids[0]

        graph = poagraph.POAGraph(init_sequence, "0")

        for i in range(len(sequences)):
            sequence = sequences[i]
            read_id = read_ids[i]

            # print("sequence\t", sequence)
            # print("label\t\t", read_id)
            # print("length\t\t", len(sequence))

            alignment = seqgraphalignment.SeqGraphAlignment(sequence, graph,
                                                            fastMethod=False,
                                                            globalAlign=True,
                                                            matchscore=1,
                                                            mismatchscore=-1,
                                                            gapscore=-2)

            # print(alignment)

            graph.incorporateSeqAlignment(alignment, sequence, str(i))

        alignments = graph.generateAlignmentStrings()

        for read_id, alignstring in alignments:
            print("{0:15s} {1:s}".format(read_id, alignstring))

        if p > 10:
            exit()
