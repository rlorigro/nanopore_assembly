from modules.pileup_utils import A,G,T,C
from handlers.FastaHandler import FastaHandler
from collections import defaultdict
from matplotlib import pyplot
import numpy


def count_runlength_per_character(sequence):
    """
    For a sequence, generate a dictionary of characters:observed_repeats, where observed_repeats is a list of n
    repeats observed for all sequential stretches of that character in the sequence
    :param sequences:
    :return:
    """
    character_counts = defaultdict(list)
    current_character = None
    current_count = 0

    for character in sequence:
        if character != current_character:
            character_counts[character].append(current_count)
            current_count = 0
        else:
            current_count += 1

        current_character = character

    return character_counts


def main():
    # output_root_dir = "output/"
    # instance_dir = "spoa_pileup_generation_" + get_current_timestamp()
    # output_dir = os.path.join(output_root_dir, instance_dir)

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

    # -------------------------------------------------------------------------

    fasta_handler = FastaHandler(reference_file_path)
    contig_names = fasta_handler.get_contig_names()

    # chromosome_name = "NC_003279.8"     # celegans chr1
    chromosome_name = "NC_003283.11"     # celegans chr5
    # chromosome_name = "1"
    # chromosome_name = "chr" + chromosome_name

    chromosome_length = fasta_handler.get_chr_sequence_length(chromosome_name)

    base_runlengths = [list() for base in [A,G,T,C]]

    reference_sequence = fasta_handler.get_sequence(chromosome_name=chromosome_name, start=0, stop=chromosome_length)

    character_counts = count_runlength_per_character(reference_sequence)

    figure, axes = pyplot.subplots(nrows=len(character_counts.keys()), sharex=True, sharey=True)

    for k,key in enumerate(character_counts.keys()):
        counts = character_counts[key]
        max_count = max(counts)

        step = 1
        bins = numpy.arange(0, max_count + step, step=step)
        frequencies, bins = numpy.histogram(counts, bins=bins, normed=False)

        print(bins.shape)
        center = (bins[:-1] + bins[1:])/2 - step/2

        axes[k].bar(center, frequencies, width=step, align="center")
        axes[k].set_ylabel(str(key))
        axes[k].set_xticks(numpy.arange(0, max_count + 1))

    pyplot.show()


if __name__ == "__main__":
    main()
