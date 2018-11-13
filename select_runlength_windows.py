from handlers.FastaHandler import FastaHandler
from handlers.FileManager import FileManager
from modules.AlignedSegmentGrabber import MAX_COVERAGE
from multiprocessing import Pool
import multiprocessing
import pickle
import pysam
import sys
import os
import gc


MIN_COVERAGE_THRESHOLD = 4                  # ignore regions with fewer than this many reads mapped
MAX_COVERAGE_THRESHOLD = MAX_COVERAGE       # only record data from up to this many reads
MAX_UNFILTERED_COVERAGE_THRESHOLD = 500     # ignore regions with more than this many reads mapped


def write_windows_to_file(windows, output_dir, filename):
    FileManager.ensure_directory_exists(output_dir)

    filename = filename + "_windows.pkl"
    path = os.path.join(output_dir, filename)

    with open(path, 'wb') as output:
        pickle.dump(windows, output, pickle.HIGHEST_PROTOCOL)


def get_transition_positions(start_position, sequence):
    positions = set()
    current_character = None

    for c,character in enumerate(sequence):
        if character != current_character:
            positions.add(start_position+c)

        current_character = character

    return positions


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


def count_matches(pileup_column, ref_character):
    n_matches = 0

    n = 0
    for pileup_read in pileup_column.pileups:
        index = pileup_read.query_position

        if index is not None:
            read_character = pileup_read.alignment.query_sequence[index]

            if read_character == ref_character:
                # match
                n_matches += 1

        n += 1

        if n > MAX_COVERAGE_THRESHOLD:
            break

    # print(n_matches)

    return n_matches, n


def find_maximum_index(a):
    max_value = -sys.maxsize
    max_index = None
    for i, value in enumerate(a):
        if value > max_value:
            max_value = value
            max_index = i

    return max_index


def select_window_edges(positions, frequencies, min_size, max_size):
    if len(positions) < 2:
        return []
    if positions[-1] - positions[0] < min_size:
        return []

    windows = list()
    i = 0
    previous_max_position = positions[0]
    max_frequency = 0.0
    max_position = 0
    max_index = i
    window_length = 0

    # j = 0
    while i < len(positions):
        window_length = positions[i] - previous_max_position

        # print(i, max_index, window_length, max_position, max_frequency)

        if min_size < window_length < max_size and i < len(positions):
            # update max frequency during window
            if frequencies[i] > max_frequency:
                max_frequency = frequencies[i]
                max_position = positions[i]
                max_index = i

        elif window_length >= max_size and i < len(positions):
            # window exceeded, save results and reset from prev max
            if previous_max_position > 0:
                windows.append([previous_max_position, max_position-1])

            # no data here or very poor quality?
            if max_frequency == 0.0:
                previous_max_position = positions[i]    # maybe skip some positions, but this prevents endless loop
            else:
                max_frequency = 0.0
                previous_max_position = max_position
                i = max_index

        elif window_length <= min_size and i < len(positions):
            pass

        elif i > len(positions):
            break

        i += 1
        # j += 1

        # if j > 10000 and i < len(frequencies):
        #     print(j, positions[0], positions[-1], len(positions), i, frequencies[i], max_index, window_length, max_position, max_frequency)

    # input list fully iterated, save results and break
    if window_length > min_size:
        windows.append([previous_max_position, max_position - 1])

    return windows


def get_column_match_frequencies(pileup_columns, reference_sequence, region, transition_positions):
    anchor_positions = list()
    anchor_match_frequencies = list()

    prev_transition_position = None
    prev_transition_n_matches = None

    prev_coverage = None
    prev_n_matches = None

    for pileup_column in pileup_columns:
        position = pileup_column.pos
        ref_character = reference_sequence[position]

        total_coverage = pileup_column.nsegments

        if total_coverage < MAX_UNFILTERED_COVERAGE_THRESHOLD:
            if region[0] < position < region[1]:
                if position in transition_positions:
                    n_matches, coverage = count_matches(pileup_column, ref_character)

                    if coverage > MIN_COVERAGE_THRESHOLD:
                        if prev_transition_position == position - 1:
                            match_frequency = (prev_transition_n_matches + n_matches) / (prev_transition_coverage + coverage)
                        elif prev_n_matches is not None:
                            match_frequency = (prev_n_matches + n_matches) / (prev_coverage + coverage)
                        else:
                            continue

                        anchor_positions.append(position)
                        anchor_match_frequencies.append(match_frequency)

                        prev_transition_position = position
                        prev_transition_n_matches = n_matches
                        prev_transition_coverage = coverage

                if position + 1 in transition_positions:
                    prev_n_matches, prev_coverage = count_matches(pileup_column, ref_character)

    return anchor_positions, anchor_match_frequencies


def select_windows(bam_file_path, chromosome_name, subregion, reference_sequence, min_size, max_size, output_dir, counter, n_chunks):
    sam_file = pysam.AlignmentFile(bam_file_path, "rb")

    pileup_columns = sam_file.pileup(chromosome_name, subregion[0], subregion[1])

    transition_positions = get_transition_positions(start_position=subregion[0],
                                                    sequence=reference_sequence[subregion[0]:subregion[1] + 1])

    # print(len(transition_positions))

    anchor_positions, anchor_match_frequencies = \
        get_column_match_frequencies(pileup_columns=pileup_columns,
                                     reference_sequence=reference_sequence,
                                     region=subregion,
                                     transition_positions=transition_positions)

    sam_file.close()

    windows = select_window_edges(positions=anchor_positions,
                                  frequencies=anchor_match_frequencies,
                                  min_size=min_size,
                                  max_size=max_size)

    filename = "_".join(map(str, subregion))

    if len(windows) > 0:
        write_windows_to_file(windows=windows,
                              output_dir=output_dir,
                              filename=filename)

    gc.collect()

    if counter is not None:
        counter.value += 1
        sys.stdout.write('\r' + "%.2f%% Completed" % (100 * counter.value / n_chunks))


def generate_argument_pools(pool_size, bam_file_path, chromosome_name, region_windows, reference_sequence, min_size, max_size, output_dir, counter, n_chunks):
    pooled_args = list()
    i = 0
    while i < len(region_windows):
        j = 0
        pool = list()
        while i + j < len(region_windows) and j < pool_size:
            subregion = region_windows[i+j]
            args = [bam_file_path, chromosome_name, subregion, reference_sequence, min_size, max_size, output_dir,
                    counter, n_chunks]
            pool.append(args)
            j += 1

        pooled_args.append(pool)
        i += j

    return pooled_args


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
    # bam_file_path = "/home/ryan/data/Nanopore/celegans/all_chips_20k_Boreal_minimap2.sorted.filtered2820.bam"
    # reference_file_path = "/home/ryan/data/Nanopore/celegans/GCF_000002985.6_WBcel235_genomic.fasta"

    # ---- Nanopore GUPPY - E. Coli - (dev machine) -------------------------
    bam_file_path = "/home/ryan/data/Nanopore/ecoli/miten/r9_ecoli_reads_vs_ref.bam"
    reference_file_path = "/home/ryan/data/Nanopore/ecoli/miten/refEcoli.fasta"

    # -------------------------------------------------------------------------

    fasta_handler = FastaHandler(reference_file_path)
    contig_names = fasta_handler.get_contig_names()
    fasta_handler.close()

    # chromosome_name = "NC_003279.8"     # celegans chr1
    # chromosome_name = "NC_003283.11"     # celegans chr5

    for chromosome_name in contig_names:
        if chromosome_name == "NC_001328.1":    # mitochondrial
            continue

        print("STARTING:", chromosome_name)
        fasta_handler = FastaHandler(reference_file_path)
        chromosome_length = fasta_handler.get_chr_sequence_length(chromosome_name)
        reference_sequence = fasta_handler.get_sequence(chromosome_name=chromosome_name,
                                                        start=0,
                                                        stop=chromosome_length)

        fasta_handler.close()

        region = [0+1000000, chromosome_length-1000000]

        max_threads = 30

        window_size = 10000
        min_size = 20
        max_size = 80

        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)

        region_windows = chunk_region(region=region, size=window_size)

        n_chunks = len(region_windows)

        print("subregions: ", n_chunks)

        output_dir = "output/window_selection/" + str(chromosome_name) + "_" + str(region[0]) + "_" + str(region[1]) + "_" + FileManager.get_datetime_string()
        print(output_dir)

        # args = list()
        # for subregion in region_windows:
        #     args.append([bam_file_path, chromosome_name, subregion, reference_sequence, min_size, max_size, output_dir, counter, n_chunks])

        pooled_args = generate_argument_pools(pool_size=max_threads,
                                              bam_file_path=bam_file_path,
                                              chromosome_name=chromosome_name,
                                              region_windows=region_windows,
                                              reference_sequence=reference_sequence,
                                              min_size=min_size,
                                              max_size=max_size,
                                              output_dir=output_dir,
                                              counter=counter,
                                              n_chunks=n_chunks)

        # print(len(pooled_args))
        # s = 0
        # for pool in pooled_args:
        #     s += len(pool)
        #     print(len(pool))
        # print(len(region_windows))
        # print(s)
        # exit()

        for arg_pool in pooled_args:
            # initiate threading
            gc.collect()
            with Pool(processes=max_threads) as pool:
                pool.starmap(select_windows, arg_pool)

    print()


if __name__ == "__main__":
    # windows = chunk_region_into_windows([0,200], size=30)
    # print(windows)
    main()
