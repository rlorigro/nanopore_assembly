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
        read_id = str(i)

        alignment = seqgraphalignment.SeqGraphAlignment(sequence, graph,
                                                        fastMethod=False,
                                                        globalAlign=True,
                                                        matchscore=1,
                                                        mismatchscore=-1,
                                                        gapscore=-2)

        graph.incorporateSeqAlignment(alignment, sequence, str(i))

    return graph


def print_test_results(sequences, test_sequence, alignments):
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
    print(alignments[-2][0], '\t', alignments[-2][1].replace("-",""))


def test_order_of_alignment_minimal():
    sequences = ['ATATTGTGTAAGGCACAATTAACA',
                 'ATATTGCAAGGCACAATTCAACA',
                 'ATATTGCAAGGCACACAACA',
                 'ATGTGCAAGAGCACATAACA']
    test_sequence = "ATATTGCAAGGCACACTAACA"

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


# It's also surprising that sequence '3' aligns its last T to the C of sequence '1'. You can see this in the graph visualization too.
#
# ```
# Aligned sequences:
# 0               ATATTGTGTAAG-GCACAATT--AACA
# 1               ATATTGC--AAG-GCACAATTC-AACA
# 2               ATATTGC--AAG-GCACA---C-AACA
# 3               ATGT-GC--AAGAGCACA---T-AACA
# ```
#
# It contains two gaps flanking the T already, so realigning the T (see below) should be the same cost, right?
#
# ```
# Aligned sequences:
# 0               ATATTGTGTAAG-GCACAATT--AACA
# 1               ATATTGC--AAG-GCACAATTC-AACA
# 2               ATATTGC--AAG-GCACA---C-AACA
# 3               ATGT-GC--AAGAGCACA-T---AACA
# ```
#
# Why does poapy create a new node and align it to C in this case?