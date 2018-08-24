import random
from poapy import poagraph
from poapy import seqgraphalignment
from itertools import combinations


def generate_identical_sequences(n_sequences, length, character_pool):
    sequence = generate_sequence(length=length, character_pool=character_pool)

    sequences = [sequence]*n_sequences

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
        exit("ERROR: query label not found in alignments")

    return query_alignment


def test_identical_sequences():
    character_pool = ["A","C","G","T"]

    sequences = generate_identical_sequences(n_sequences=6, length=10, character_pool=character_pool)

    print("Sequences before alignment:")
    for s, sequence in enumerate(sequences):
        print(s, sequence)

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

    alignments = graph.generateAlignmentStrings()

    print("\nAligned sequences:")
    for read_id, alignstring in alignments:
        print("{0:15s} {1:s}".format(read_id, alignstring))


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


def test_order_of_alignment():
    # test_sequence = "ATATTGCAAGGCACACTAACA"
    # sequences = ["ATATTGTGTAAGGCACAATTAACA",
    #              "ATATTGCAAGGCACACTAACA",
    #              "ATATTGCAAGGCACAATTCAACA",
    #              "ATATTACAGAGCACACTA",
    #              "GTATTGCAAGCAAGCACACAACAA",
    #              "ATATTGCAAGGCACACAACA",
    #              "ATATTGCAAGGCACACTAACAATAA",
    #              "ATATTTACAGGAGCACACACTAACA",
    #              "ATATTGCAAGGCACACTAACA",
    #              "ATGTGCAAAGACACACTAACCA",
    #              "CTGTTACAAAAACTTTTAACA",
    #              "ATATTGCAAGACACACTAACA",
    #              "ATGTGCAAGAGCACATAACA",
    #              "ATATTTTAAGGCACACTAACA",
    #              "ATATTGCAAGGCACACTAACA",
    #              "ATGTTACAGAGCACACTAACA",
    #              "ATATTGCAAGGCATA",
    #              "ATACTGTAAGGCACACTTAAACAACA",
    #              "ATATTGTAAGGCACACTTCAACA",
    #              "ATATTGCAAGGCACACTAACA",
    #              "ATATACGTAAGGCACACTAAATA",
    #              "ATATTGCACACTAACA",
    #              "AATTAAAAAGGCATACTCCAATAA",
    #              "ATATTGCAAGGCCACACTGGTCAA",
    #              "ATGTTGCAAGGCACTAACA",
    #              "ATGTTGCAGGCACACTATA",
    #              "ATATTGCAAGGCACACTAACA",
    #              "ATATTGCAAAGCACACTAACA",
    #              "ATATTGCAAGGCACAATTAACA",
    #              "ATGTTACAAGACACACTAATATAAA",
    #              "ATTTACAAACACTAACA",
    #              "ATATTGCAGAGCACTAACA",
    #              "ATATTGCAAGGCACACTCAACA",
    #              "ATATTGCAAGGCACACTAACA",
    #              "ATATTGCAAGGCACACTAATA"]

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

    # sequences = ['TGGAATGAAAGAGGAAAAGAA', 'AGGAATGAAAGAGGAAAAGAA', 'TGGAATGAAAGAGAAAAGAA', 'TGAGATAAAGAAGAGGAAAAGAA',
    #  'TGGAATGAAAGAGGAAA', 'TGGAATGAAAGAGTGAAAAGAA', 'TGGAAATGAAAGAGGGAAAAGAA', 'TAGTGAAAGAGGAAAAGAA',
    #  'TGGAAGAAGAGGAAAAGAA', 'TGGGAATGAAAGGAGGAAAGAA', 'TGGAATATAGAGAAGAGGAAAAAGAAA', 'TGGAAATGAAAGAGGAAAAGAA']
    #
    # test_sequence = "GCCCAGAAATTCCAGACCAGC"

    minimum_error_set = sequences
    for i in range(1,len(sequences)):
        print(i)
        for combination in combinations(sequences, i):
            graph = generate_poa_graph(combination)

            alignment = seqgraphalignment.SeqGraphAlignment(test_sequence, graph,
                                                            fastMethod=False,
                                                            globalAlign=True,
                                                            matchscore=1,
                                                            mismatchscore=-1,
                                                            gapscore=-2)

            graph.incorporateSeqAlignment(alignment, test_sequence, "test")

            alignments = graph.generateAlignmentStrings()
            test_alignment = get_alignment_by_label(alignments, "test")

            if test_alignment[1].replace("-","") != test_sequence:
                minimum_error_set = combination
                break

        if test_alignment[1].replace("-", "") != test_sequence:
            print("before alignment")
            print("test", test_sequence)
            print("after alignment")
            print(test_alignment[0], test_alignment[1].replace("-", ""))

            minimum_error_set = combination
            break

    print(minimum_error_set)

    # print("\nAligned sequences:")
    # for read_id, alignstring in alignments:
    #     print("{0:15s} {1:s}".format(read_id, alignstring))


def test_order_of_alignment_minimal(sequences, test_sequence):
    # sequences = ['ATATTGTGTAAGGCACAATTAACA',
    #              'ATATTGCAAGGCACAATTCAACA',
    #              'ATATTGCAAGGCACACAACA',
    #              'ATGTGCAAGAGCACATAACA']
    # test_sequence = "ATATTGCAAGGCACACTAACA"

    # sequences = ['CTACTTGGGAGGCTGAGGTGG', 'CCACTTGAGTTGAGG', 'CTACTTGGGAAGCTAGAGGTGG']
    # test_sequence = "CTACTTGGGAGGCTGAGGTGG"

    sequences = ["TAGTGAAAGAGGAAAAGAA"]
    test_sequence = "GCCCAGAAATTCCAGACCAGC"

    graph = generate_poa_graph(sequences)

    alignment = seqgraphalignment.SeqGraphAlignment(test_sequence, graph,
                                                    fastMethod=False,
                                                    globalAlign=True,
                                                    matchscore=1,
                                                    mismatchscore=-1,
                                                    gapscore=-2)

    graph.incorporateSeqAlignment(alignment, test_sequence, "test")

    alignments = graph.generateAlignmentStrings()

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

    with open("test_poapy.html", 'w') as file:
        graph.htmlOutput(file)


if __name__ == "__main__":
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

    sequences = ['CTACTTGGGAGGCTGAGGTGG', 'CCACTTGAGTTGAGG', 'CTACTTGGGAAGCTAGAGGTGG']
    test_sequence = "CTACTTGGGAGGCTGAGGTGG"

    test_order_of_alignment_minimal(sequences=sequences, test_sequence=test_sequence)

    sequences = ["TAGTGAAAGAGGAAAAGAA"]
    test_sequence = "GCCCAGAAATTCCAGACCAGC"

    test_order_of_alignment_minimal(sequences=sequences, test_sequence=test_sequence)

    sequences = ["TTA", "TGC"]
    test_sequence = "TTGC"

    test_order_of_alignment_minimal(sequences=sequences, test_sequence=test_sequence)

    test_identical_sequences()


    # test_order_of_alignment()
