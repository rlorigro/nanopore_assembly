from generate_spoa_pileups_from_bam import convert_alignments_to_matrix, convert_aligned_reference_to_one_hot, sequence_to_index
from modules.IterativeHistogram import IterativeHistogram
import numpy


class ConsensusCaller:
    def __init__(self, sequence_to_index):
        self.index_to_sequence = self.get_inverse_dictionary(sequence_to_index)
        self.histogram = IterativeHistogram(start=-0.1, stop=1.1, n_bins=6)

    def get_inverse_dictionary(self, a):
        inverse = dict()
        for item in a.items():
            inverse[item[1]] = item[0]

        return inverse

    def call_consensus_as_one_hot(self, pileup_matrix):
        """
        Create a binarized one hot of the consensus, with dimensions [5,c] where c is the number of columns in pileup,
        and 5 is the number of symbols allowed: [-,A,G,T,C]
        :param pileup_matrix:
        :param format:
        :return:
        """
        n, m = pileup_matrix.shape

        one_hot = numpy.zeros([5, m])

        for column_index in range(m):
            column = pileup_matrix[:, column_index]
            unique, inverse = numpy.unique(column, return_inverse=True)
            bincount = numpy.bincount(inverse)
            mode = bincount.argmax()
            consensus_float = unique[mode]
            character_index = self.histogram.get_bin(consensus_float)

            one_hot[character_index, column_index] = 1

        return one_hot

    def call_consensus_as_string(self, pileup_matrix):
        """
        Create a string of the consensus at each column
        :param pileup_matrix:
        :param format:
        :return:
        """
        n, m = pileup_matrix.shape

        consensus_characters = list()

        for column_index in range(m):
            column = pileup_matrix[:, column_index]
            unique, inverse = numpy.unique(column, return_inverse=True)
            bincount = numpy.bincount(inverse)
            mode = bincount.argmax()
            consensus_float = unique[mode]
            character_index = self.histogram.get_bin(consensus_float)
            character = self.index_to_sequence[character_index]

            consensus_characters.append(character)

        consensus_string = ''.join(consensus_characters)
        return consensus_string

    def call_consensus_as_encoding(self, pileup_matrix):
        """
        Create a vector of float encodings from 0-1 for each nucleotide
        :param pileup_matrix:
        :param format:
        :return:
        """
        n, m = pileup_matrix.shape

        encoding = numpy.zeros([1, m])

        for column_index in range(m):
            column = pileup_matrix[:, column_index]
            unique, inverse = numpy.unique(column, return_inverse=True)
            bincount = numpy.bincount(inverse)
            mode = bincount.argmax()
            consensus_float = unique[mode]

            encoding[0, column_index] = round(consensus_float, 3)

        return encoding

    def call_consensus_as_normalized_frequencies(self, pileup_matrix):
        """
        Create a binarized one hot of the consensus, with dimensions [5,c] where c is the number of columns in pileup,
        and 5 is the number of symbols allowed: [-,A,G,T,C]
        :param pileup_matrix:
        :param format:
        :return:
        """
        n, m = pileup_matrix.shape

        one_hot = numpy.zeros([5, m])

        for column_index in range(m):
            column = pileup_matrix[:, column_index]
            unique, inverse = numpy.unique(column, return_inverse=True)
            bincount = numpy.bincount(inverse)

            n_characters = unique.shape[0]
            for i in range(n_characters):
                consensus_float = unique[i]
                character_index = self.histogram.get_bin(consensus_float)
                frequency = bincount[i]

                one_hot[character_index, column_index] = frequency

        # normalize
        column_sums = numpy.sum(one_hot, axis=0)
        one_hot = one_hot/column_sums

        return one_hot


def test_consensus_caller():
    alignments = [["0",  "TTTG--T--TG-C-------T-GCTATCG-----CC----G--T"],
                  ["1",  "TTTG--T--TGCC-----ATT-GCT-T-T-----TC----G--T"],
                  ["2",  "TTTG--T--TGCC-----ATT-GCT-T-TC-G--T-----G--T"],
                  ["3",  "-TTG---GTTGCC-----A-T-G-T-T-T-----T----GG--T"],
                  ["4",  "-TTG---GTTGCC-----ATT-GC--T-T-----TC----G--T"],
                  ["5",  "TTTG--T--TGCC-----ATT-GC--T-G------C-------T"],
                  ["6",  "TTTG--T--TG-C-T--TATT-GC----T-----TC----GA-T"],
                  ["7",  "TTTG--T--TGCC-----ATT-GCT-T-TC----TC--G-G--T"],
                  ["8",  "TTTG--T--TGCC-----ATT-GCT-T-TCGGTAT-----G--T"],
                  ["9",  "TTTG--T--TGCC-----ATT-G-----------T-----G--T"],
                  ["10", "TTTG--T--TGCC-----ATT-GC--T-T-----TC----G--T"],
                  ["11", "TTTG--T--TGCC-----ATT-GCT-T-TC----TC----G--T"],
                  ["12", "TTTG--T--TGCC-TGAAATTAGC--T-T-----T----GG--T"],
                  ["13", "TTTG--T--TGCC-----ATT-G-T-T-T-----TC-GG-G--T"],
                  ["14", "-TTTTTTG-TGCC-----ATT-GC--T-T-----T-----GA-T"],
                  ["15", "-TTGCTT--TGCC-----ATT-GC--T-TC----T-----G--C"],
                  ["16", "TTTG--T--TGCC-----ATTAGC--T-T-----T----GG--T"],
                  ["17", "TTTGC----TG-C-----ATT-GCT-T-T-----TC----G--T"],
                  ["18", "TTTG--T--TGCCA----ATT-GC--T-T-----T-----GATT"],
                  ["19", "TTTG--T--TGCC-----ATTAGCT-T-------CC--G-G--T"],
                  ["20", "TTTG--T--TGCC-----ATT-GC----T-----TCA-G-G--T"],
                  ["21", "TTTG--T--TGCC-----ATT-GCT-T-TC-G--T------A--"]]

    ref_alignment = ["ref", "TTTG--T--TGCC-----ATT-GC--T-T-----TC--G-G--T"]

    pileup_matrix = convert_alignments_to_matrix(alignments, fixed_coverage=False)
    reference_matrix = convert_alignments_to_matrix([ref_alignment], fixed_coverage=False)
    reference_one_hot = convert_aligned_reference_to_one_hot([ref_alignment])

    consensus_caller = ConsensusCaller(sequence_to_index)

    consensus_string = consensus_caller.call_consensus_as_string(pileup_matrix)
    # consensus_one_hot = consensus_caller.call_consensus_as_one_hot(pileup_matrix)
    consensus_normalized_frequency = consensus_caller.call_consensus_as_normalized_frequencies(pileup_matrix)
    # consensus_encoding = consensus_caller.call_consensus_as_encoding(pileup_matrix)

    print(ref_alignment[1])
    print(consensus_string)

    # print(reference_matrix)
    # print(consensus_encoding)

    # print(reference_one_hot)
    # print(consensus_one_hot)

    # print(numpy.argmax(reference_one_hot, axis=0))
    # print(numpy.argmax(consensus_normalized_frequency, axis=0))

    print(reference_one_hot)
    print(consensus_normalized_frequency)


if __name__ == "__main__":
    test_consensus_caller()