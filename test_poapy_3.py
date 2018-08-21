from poapy import poagraph
from poapy import seqgraphalignment


def generate_poa_graph(sequences):
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
                                                        fastMethod=False,
                                                        globalAlign=True,
                                                        matchscore=1,
                                                        mismatchscore=-1,
                                                        gapscore=-2)

        graph.incorporateSeqAlignment(alignment, sequence, label)

    return graph


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
        exit("ERROR: query label not found in alignments")

    return query_alignment


def print_test_results(sequences, test_sequence, alignments):
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


def test_order_of_alignment_minimal():
    sequences = ["TTA", "TGC"]
    test_sequence = "TTGC"

    graph = generate_poa_graph(sequences)

    alignment = seqgraphalignment.SeqGraphAlignment(test_sequence, graph,
                                                    fastMethod=False,
                                                    globalAlign=True,
                                                    matchscore=1,
                                                    mismatchscore=-1,
                                                    gapscore=-2)

    graph.incorporateSeqAlignment(alignment, test_sequence, "test")

    alignments = graph.generateAlignmentStrings()

    print_test_results(sequences, test_sequence, alignments)

    with open("test_poapy.html", 'w') as file:
        graph.htmlOutput(file)


if __name__ == "__main__":
    test_order_of_alignment_minimal()
