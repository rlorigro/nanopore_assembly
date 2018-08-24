import random
from poapy import poagraph
from poapy import seqgraphalignment
from itertools import combinations


def get_alignments_by_sequence(alignments, sequence):
    query_alignments = list()

    for a, alignment in enumerate(alignments):
        current_sequence = alignment[1].replace("-",'')

        if current_sequence == sequence:
            query_alignments.append(alignment)

    if len(query_alignments) == 0:
        raise KeyError("ERROR: query sequence not found in alignments")

    return query_alignments


def generate_poa_graph(sequences, fastMethod=True, globalAlign=True, matchscore=1, mismatchscore=-1, gapscore=-2):
    """
    Initialize graph and align all sequences
    :param sequences:
    :return: graph: the completed POA graph resulting from the given sequences
    """
    init_sequence = sequences[0]
    init_label = "0"

    graph = poagraph.POAGraph(init_sequence, init_label)

    for i in range(1, len(sequences)):
        sequence = sequences[i]
        label = str(i)

        alignment = seqgraphalignment.SeqGraphAlignment(sequence, graph,
                                                        fastMethod=fastMethod,
                                                        globalAlign=globalAlign,
                                                        matchscore=matchscore,
                                                        mismatchscore=mismatchscore,
                                                        gapscore=gapscore)

        graph.incorporateSeqAlignment(alignment, sequence, label)

    return graph


def test_identical_sequence_subset(sequences, test_sequence, print_result=False):
    fastMethod = True
    globalAlign = True
    matchscore = 1
    mismatchscore = -1
    gapscore = -2

    graph = generate_poa_graph(sequences,
                               fastMethod=fastMethod,
                               globalAlign=globalAlign,
                               matchscore=matchscore,
                               mismatchscore=mismatchscore,
                               gapscore=gapscore)

    alignments = graph.generateAlignmentStrings()

    test_alignments = get_alignments_by_sequence(alignments=alignments, sequence=test_sequence)

    test_alignment_strings = [test_alignment[1] for test_alignment in test_alignments]

    alignment_string_set = set(test_alignment_strings)

    if print_result:
        print("\nAligned sequences:")
        for read_id, alignstring in alignments:
            print("{0:15s} {1:s}".format(read_id, alignstring))

        print("\nIdentical sequences after alignment:")
        for string in alignment_string_set:
            print(string)

    assert len(alignment_string_set) == 1


def find_minimum_error_set(sequences, test_sequence):
    for n_sequences in range(1,len(sequences)):
        for combination in combinations(sequences, n_sequences):
            try:
                test_identical_sequence_subset(sequences=combination, test_sequence=test_sequence, print_result=False)

            except KeyError:
                continue

            except AssertionError:
                return combination

    return


if __name__ == "__main__":
    sequences = ["ACTGGACATTAGAA",
                 "ACTGGTATAGCCACGTTAGA",
                 "ACTGGTATAGCCATCAAAGAA",
                 "TCAGAGAAATACGGTGAA",
                 "ACAGGTATGGCGTTTGAGAA",
                 "ACTGGTATAGCCAATGCTAGAAA",
                 "ACTGGTATAGCTAGGGGA",
                 "ACTGGTATAGCCACGTTAAGAA",
                 "ACTGGCATAGCCACGTTAGAA",
                 "ACTGGGTATAACCACGTTAGA",
                 "ATTGGTATATAGCTTTTACTAGGA",
                 "ACTGGTAATAGCCACGTTAGAA",
                 "AGATAAGCCAGATTTGAA",
                 "ACTGGTATAGCCACGTTAGA",
                 "ACTGGTATAGCCAATGCTAGAA",
                 "ACTGGTATAGCCAATGCTAGGA",
                 "ACAGGTATAGCAGTTAGAA",
                 "ACTGGTATAGCCTAATGCTTAGAA",
                 "ACTGGTAGCCATTTTCAATGCTAGAAA"]

    test_sequence = "ACTGGTATAGCCACGTTAGA"

    minimum_error_set = find_minimum_error_set(sequences=sequences, test_sequence=test_sequence)

    print("minimal error set:")
    print(minimum_error_set)

    test_identical_sequence_subset(sequences=minimum_error_set, test_sequence=test_sequence, print_result=True)
