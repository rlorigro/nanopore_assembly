from handlers.FastaHandler import FastaHandler
from modules.window_selection_utils import *
from multiprocessing import Pool
import multiprocessing
import pysam
import pickle
from datetime import datetime


def write_windows_to_file(windows, output_dir, filename):
    FileManager.ensure_directory_exists(output_dir)

    filename = filename + "_windows.pkl"
    path = os.path.join(output_dir, filename)

    with open(path, 'wb') as output:
        pickle.dump(windows, output, pickle.HIGHEST_PROTOCOL)

    # with open(path, 'rb') as input:
    #     test = pickle.load(input)
    #
    #     print(test)


def merge_windows(windows_a, windows_b):
    if len(windows_b) > 0:

        last_anchor_a = windows_a[-1][-1] + 1
        first_anchor_b = windows_b[0][0] - 1

        if first_anchor_b - last_anchor_a > 1:
            bridge_window = [last_anchor_a, first_anchor_b]

            windows = windows_a + [bridge_window] + windows_b

    else:
        windows = windows_a + windows_b

    return windows


def get_window_edges(bam_file_path, reference_sequence, chromosome_name, region_window, p_threshold, n_steps, output_dir, counter=None, n_chunks=1):
    sam_file = pysam.AlignmentFile(bam_file_path, "rb")

    intervals = chunk_region(region=region_window, size=500)

    all_windows = list()

    for interval in intervals:
        column_frequencies = get_column_frequencies(sam_file, reference_sequence, chromosome_name, interval)

        column_frequencies = numpy.concatenate(column_frequencies, axis=1)
        match_frequencies = column_frequencies[0, :]

        kernel_sums = get_kernel_sums(match_frequencies)
        pdf, cdf, bins = approximate_pdf_and_cdf(array=kernel_sums, n_steps=n_steps)

        threshold = get_threshold(cdf=cdf, p=p_threshold, n_steps=n_steps)

        passing_indices = (kernel_sums > threshold)

        windows = select_window_edges(passing_indices=passing_indices, reference_start_position=interval[0])

        filename = "_".join(map(str, interval))

        if len(all_windows) > 0:
            all_windows = merge_windows(windows_a=all_windows, windows_b=windows)

        else:
            all_windows = windows

            plot_kernel_distribution(pdf=pdf,
                                     cdf=cdf,
                                     bins=bins,
                                     save=True,
                                     output_dir=output_dir,
                                     filename=filename)

            plot_kernels_and_column_frequencies(kernel_sums=kernel_sums,
                                                passing_indices=passing_indices,
                                                column_frequencies=column_frequencies,
                                                slice_range=[100, 200],
                                                save=True,
                                                output_dir=output_dir,
                                                filename=filename)

    filename = "_".join(map(str, region_window))
    write_windows_to_file(windows=all_windows,
                          output_dir=output_dir,
                          filename=filename)

    sam_file.close()

    if counter is not None:
        counter.value += 1
        sys.stdout.write('\r' + "%.2f%% Completed" % (100 * counter.value / n_chunks))


def chunk_region(region, size):
    windows = list()
    position = region[0]
    end = 0

    while end <= region[1] - 1:
        start = position
        position = position + size
        end = position - 1

        if end >= region[1] - 1:
            end = region[1]

        windows.append([start, end])

    return windows


def main():
    # output_root_dir = "output/"
    # instance_dir = "spoa_pileup_generation_" + get_current_timestamp()
    # output_dir = os.path.join(output_root_dir, instance_dir)

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

    region = [0, chromosome_length]
    window_size = 10000

    max_threads = 30
    n_steps = 200
    p_threshold = 0.90

    now = datetime.now()
    now = [now.year, now.month, now.day, now.hour, now.minute]
    datetime_string = "_".join(list(map(str, now)))

    output_dir = "output/window_selection/" + str(chromosome_name) + "_" + str(region[0]) + "_" + str(region[1]) + "_" + datetime_string
    print(output_dir)

    reference_sequence = fasta_handler.get_sequence(chromosome_name=chromosome_name,
                                                    start=0,
                                                    stop=chromosome_length)

    parse_region_parallel(bam_file_path=bam_file_path,
                          reference_sequence=reference_sequence,
                          chromosome_name=chromosome_name,
                          region=region,
                          window_size=window_size,
                          p_threshold=p_threshold,
                          output_dir=output_dir,
                          n_steps=n_steps,
                          max_threads=max_threads)


def parse_region_parallel(bam_file_path, reference_sequence, chromosome_name, region, window_size, p_threshold, output_dir, n_steps, max_threads):
    region_windows = chunk_region(region=region, size=window_size)

    print(len(region_windows))

    n_chunks = len(region_windows)

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)

    args_per_thread = list()
    for region_window in region_windows:
        args = [bam_file_path, reference_sequence, chromosome_name, region_window, p_threshold, n_steps, output_dir, counter, n_chunks]
        args_per_thread.append(args)

    if len(args_per_thread) < max_threads:
        max_threads = len(args_per_thread)

    # initiate threading
    with Pool(processes=max_threads) as pool:
        pool.starmap(get_window_edges, args_per_thread)

    print()


if __name__ == "__main__":
    # windows = chunk_region_into_windows([0,200], size=30)
    # print(windows)

    main()
