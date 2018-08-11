import random
from poapy import poagraph
from poapy import seqgraphalignment


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


if __name__ == "__main__":
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
        read_id = str(i)

        alignment = seqgraphalignment.SeqGraphAlignment(sequence, graph,
                                                        fastMethod=False,
                                                        globalAlign=True,
                                                        matchscore=1,
                                                        mismatchscore=-1,
                                                        gapscore=-2)

        graph.incorporateSeqAlignment(alignment, sequence, str(i))

    alignments = graph.generateAlignmentStrings()

    print("\nAligned sequences:")
    for read_id, alignstring in alignments:
        print("{0:15s} {1:s}".format(read_id, alignstring))
