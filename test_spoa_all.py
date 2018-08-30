from subprocess import Popen, PIPE
import random


GLOBAL_ALIGN = True
MATCH_SCORE = 1
MISMATCH_SCORE = -1
GAP_SCORE = -2


def call_commandline_spoa(space_separated_sequences, ref_sequence):
    match_arg = str(MATCH_SCORE)
    mismatch_arg = str(MISMATCH_SCORE)
    gap_arg = str(GAP_SCORE)
    alignment_arg = str(1) if GLOBAL_ALIGN else str(0)

    args = ["/home/ryan/software/spoa/align_sequences_and_reference",
            alignment_arg,
            match_arg,
            mismatch_arg,
            gap_arg,
            space_separated_sequences,
            ref_sequence]

    process = Popen(args,
                    stdout=PIPE,
                    stderr=PIPE)

    stdout, stderr = process.communicate()

    stdout = stdout.decode("UTF-8").strip()
    stderr = stderr.decode('UTF-8').strip()

    alignment_strings = stdout.split('\n')
    read_alignment_strings = alignment_strings[:-1]
    ref_alignment_string = alignment_strings[-1]

    if stderr != "":
        exit(stderr)

    return read_alignment_strings, ref_alignment_string


def call_commandline_spoa_no_ref(space_separated_sequences):
    match_arg = str(MATCH_SCORE)
    mismatch_arg = str(MISMATCH_SCORE)
    gap_arg = str(GAP_SCORE)
    alignment_arg = str(1) if GLOBAL_ALIGN else str(0)

    args = ["/home/ryan/software/spoa/align_sequences",
            alignment_arg,
            match_arg,
            mismatch_arg,
            gap_arg,
            space_separated_sequences]

    process = Popen(args,
                    stdout=PIPE,
                    stderr=PIPE)

    stdout, stderr = process.communicate()

    stdout = stdout.decode("UTF-8").strip()
    stderr = stderr.decode('UTF-8').strip()

    alignment_strings = stdout.split('\n')

    if stderr != "":
        exit(stderr)

    return alignment_strings


def get_spoa_alignment_no_ref(sequences):
    space_separated_sequences = ' '.join(sequences)

    read_alignment_strings = call_commandline_spoa_no_ref(space_separated_sequences)

    alignments = list()
    for a, alignment_string in enumerate(read_alignment_strings):
        label = str(a)
        alignment = [label, alignment_string]

        alignments.append(alignment)

    return alignments


def get_spoa_alignment(sequences, ref_sequence):
    space_separated_sequences = ' '.join(sequences)

    read_alignment_strings, ref_alignment_string = call_commandline_spoa(space_separated_sequences, ref_sequence)

    alignments = list()
    for a, alignment_string in enumerate(read_alignment_strings):
        label = str(a)
        alignment = [label, alignment_string]

        alignments.append(alignment)

    ref_label = "ref"
    ref_alignment = [ref_label, ref_alignment_string]

    return alignments, ref_alignment


def print_order_of_alignment_results(sequences, test_sequence, alignments):
    test_alignment = get_alignment_by_label(alignments, "test")

    print("Unaligned sequences:")
    for sequence in sequences:
        print(sequence)

    print("\nUnaligned test sequence:")
    print(test_sequence)

    print("\nAligned sequences:")
    for read_id, alignstring in alignments:
        print("{0:15s} {1:s}".format(read_id, alignstring))

    print("\nTest before alignment")
    print("test\t", test_sequence)
    print("Test after alignment")
    print(test_alignment[0], '\t', test_alignment[1].replace("-",""))


def print_identical_subset_result(sequences, alignments, alignment_string_set):
    print("Unaligned sequences:")
    for sequence in sequences:
        print(sequence)

    print("\nAligned sequences:")
    for read_id, alignstring in alignments:
        print("{0:15s} {1:s}".format(read_id, alignstring))

    for string in alignment_string_set:
        print(string)


def generate_identical_sequences(n_sequences, length, character_pool):
    sequence = generate_sequence(length=length, character_pool=character_pool)

    sequences = [sequence] * n_sequences

    return sequences


def generate_sequence(length, character_pool):
    sequence = list()

    for i in range(length):
        sequence.append(random.choice(character_pool))

    sequence_string = ''.join(sequence)

    return sequence_string


def get_alignment_by_label(alignments, label):
    """
    iterate through poapy alignment tuples to find an alignment with the specified label
    :param alignments:
    :return:
    """
    query_alignment = None

    for alignment in alignments:
        if alignment[0] == label:
            query_alignment = alignment

    if query_alignment is None:
        raise KeyError("ERROR: query sequence not found in alignments")

    return query_alignment


def get_alignments_by_sequence(alignments, sequence):
    """
    iterate through poapy alignment tuples to find all alignments with the specified sequence
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


def test_identical_sequences(print_result=False):
    """
    Randomly generate identical DNA sequences and ensure that their aligned strings are identical
    :param print_result:
    :return:
    """
    character_pool = ["A", "C", "G", "T"]

    sequences = generate_identical_sequences(n_sequences=6, length=10, character_pool=character_pool)

    alignments = get_spoa_alignment_no_ref(sequences=sequences)

    alignment_strings = [alignment[1] for alignment in alignments]
    alignment_string_set = set(alignment_strings)

    assert len(alignment_string_set) == 1

    if print_result:
        print("Sequences before alignment:")
        for s, sequence in enumerate(sequences):
            print(s, sequence)

        print("\nAligned sequences:")
        for read_id, alignstring in alignments:
            print("{0:15s} {1:s}".format(read_id, alignstring))


def test_order_of_alignment_minimal(sequences, test_sequence, print_result=False):
    """
    Assert that the given test sequence retains its order after alignment
    :param sequences:
    :param test_sequence:
    :param print_result:
    :return:
    """
    ref_sequence = test_sequence

    alignments, ref_alignment = get_spoa_alignment(sequences=sequences, ref_sequence=ref_sequence)

    alignments.append(ref_alignment)

    test_alignment = get_alignment_by_label(alignments=alignments, label="ref")
    aligned_test_sequence = test_alignment[1].replace("-","")

    if print_result:
        print_order_of_alignment_results(sequences=sequences, test_sequence=test_sequence, alignments=alignments)

    assert aligned_test_sequence == test_sequence


def test_identical_sequence_subset(sequences, test_sequence, print_result=False):
    """
    Test case in which certain combinations of sequences cause identical sequences to be aligned differently
    :param sequences:
    :param test_sequence:
    :param print_result:
    :return:
    """
    fastMethod = True
    globalAlign = True
    matchscore = 4
    mismatchscore = -4
    gapscore = -5

    alignments = get_spoa_alignment_no_ref(sequences=sequences)

    test_alignments = get_alignments_by_sequence(alignments=alignments, sequence=test_sequence)

    test_alignment_strings = [test_alignment[1] for test_alignment in test_alignments]

    test_alignment_string_set = set(test_alignment_strings)

    if print_result:
        print_identical_subset_result(sequences=sequences,
                                      alignments=alignments,
                                      alignment_string_set=test_alignment_string_set)

    assert len(test_alignment_string_set) == 1


def test_all_cases():
    print("Order of alignment case 1:")

    sequences = ['CTACTTGGGAGGCTGGAGGTGG',
                 'CTACTTGGGAGGCTGAGGTGG',
                 'CTACTTGGGAGGCTGAGGGGGTGG',
                 'CTACTTGGGAGGCTGGGGTGG',
                 'CTACTTGGGAGGCTGGGAGGTGG',
                 'CTACTTGGGAGGCTGAGGTGG',
                 'CTACTTGGGAGGCTGAGGTGG',
                 'CTACTTTGGGAGGCTGAGGTGG',
                 'CCACTTGAGTTGAGG',
                 'CTACTTGGGAAGCTAGAGGTGG',
                 'ATACTTAGGAGGCTGAGGTGG',
                 'CCACTTTGGGAGGCTGAGGG']

    test_sequence = "CTACTTGGGAGGCTGAGGTGG"

    test_order_of_alignment_minimal(sequences=sequences, test_sequence=test_sequence)

    print("PASS")
    print("Order of alignment case 2:")

    sequences = ['CTACTTGGGAGGCTGAGGTGG', 'CCACTTGAGTTGAGG', 'CTACTTGGGAAGCTAGAGGTGG']
    test_sequence = "CTACTTGGGAGGCTGAGGTGG"

    test_order_of_alignment_minimal(sequences=sequences, test_sequence=test_sequence)

    print("PASS")
    print("Order of alignment case 3:")

    sequences = ["TAGTGAAAGAGGAAAAGAA"]
    test_sequence = "GCCCAGAAATTCCAGACCAGC"

    test_order_of_alignment_minimal(sequences=sequences, test_sequence=test_sequence)

    print("PASS")
    print("Order of alignment case 4:")

    sequences = ["TTA", "TGC"]
    test_sequence = "TTGC"

    test_order_of_alignment_minimal(sequences=sequences, test_sequence=test_sequence)

    print("PASS")
    print("Identical sequences:")

    test_identical_sequences()

    print("PASS")
    print("Identical sequence subset:")

    sequences = ['TCTTTATCCATTC', 'TCTTGGTTCATTTCATGCTCG', 'TCTTTGTCCATTTCATGCTTC', 'TCTTTGTCCATTTCATGCTTC']

    test_sequence = "TCTTTGTCCATTTCATGCTTC"

    test_identical_sequence_subset(sequences=sequences, test_sequence=test_sequence)

    print("PASS")

    sequences = ["ATATTGTTTATCATGTTTAAA",
                 "ATATTGTTTATCATGTTTAAA",
                 "ATATTGTTGATTATGTTTAAACA",
                 "ATATTGTTTATCATGTTTAAA",
                 "ATATTATTGTTTATCATGTTTAAA",
                 "ATATTGTTTATCATGTTTAAA",
                 "ATATTGTTTTATATCATGTTTAAA",
                 "ATATTGTTTATCATGTTTAAAA",
                 "ATATTGTTTATCATGTTTAAA",
                 "ATATTGTTTATCGCGTTTAAA",
                 "ATATTGTTTATCATGTTTAAAA",
                 "ATATTGTTTATCATGTTTAAA",
                 "ATATTGTTTATCATGTTTAAA",
                 "ATATTGTTTGCTGAA",
                 "ATATTGTTTATCTATGTTTAAA",
                 "ATATTGTCTACTCATAAGTTTGAA",
                 "ATATTGTTTATCATGTTTAAA",
                 "ATATTGTTTATCATGTTTAAAA",
                 "ATATTGTTTATCGTTTAAA",
                 "ATATTGTTTATCATGTTTTAAA"]

    test_sequence = "ATATTGTTTATCATGTTTAAA"

    test_identical_sequence_subset(sequences=sequences, test_sequence=test_sequence)

    print("PASS")


if __name__ == "__main__":
    test_all_cases()