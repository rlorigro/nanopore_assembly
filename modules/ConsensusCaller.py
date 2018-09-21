import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from modules.pileup_utils import convert_collapsed_alignments_to_matrix
from modules.pileup_utils import convert_aligned_reference_to_one_hot
from modules.pileup_utils import convert_alignments_to_matrix
from modules.pileup_utils import sequence_to_index, sequence_to_float
from matplotlib import pyplot
import numpy


class ConsensusCaller:
    def __init__(self, sequence_to_index, sequence_to_float):
        self.index_to_sequence = self.get_inverse_dictionary(sequence_to_index)
        self.float_to_index = self.get_float_to_index(sequence_to_float, sequence_to_index)

    def get_inverse_dictionary(self, a):
        inverse = dict()
        for item in a.items():
            inverse[item[1]] = item[0]

        return inverse

    def get_float_to_index(self, sequence_to_float, sequence_to_index):
        float_to_index = dict()

        for sequence, float_value in sequence_to_float.items():
            float_to_index[float_value] = sequence_to_index[sequence]

        return float_to_index

    def call_consensus_as_one_hot(self, pileup_matrix):
        """
        Create a binarized one hot of the consensus, with dimensions [5,c] where c is the number of columns in pileup,
        and 5 is the number of symbols allowed: [-,A,G,T,C]
        :param pileup_matrix:
        :param format:
        :return:
        """
        pileup_matrix = pileup_matrix.round(3)
        pileup_matrix = numpy.atleast_2d(pileup_matrix)

        n, m = pileup_matrix.shape

        one_hot = numpy.zeros([5, m])

        for column_index in range(m):
            column = pileup_matrix[:, column_index]
            unique, inverse = numpy.unique(column, return_inverse=True)
            bincount = numpy.bincount(inverse)
            mode = bincount.argmax()
            consensus_code = round(float(unique[mode]),3)

            # print(column)
            # print(unique)
            # print(bincount)
            # print(consensus_code)

            character_index = self.float_to_index[consensus_code]

            one_hot[character_index, column_index] = 1

        return one_hot

    def call_consensus_as_string(self, pileup_matrix):
        """
        Create a string of the consensus at each column
        :param pileup_matrix:
        :param format:
        :return:
        """
        pileup_matrix = pileup_matrix.round(3)

        n, m = pileup_matrix.shape

        consensus_characters = list()

        for column_index in range(m):
            column = pileup_matrix[:, column_index]
            unique, inverse = numpy.unique(column, return_inverse=True)
            bincount = numpy.bincount(inverse)
            mode = bincount.argmax()
            consensus_code = unique[mode]
            character_index = self.float_to_index[consensus_code]
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
        pileup_matrix = pileup_matrix.round(3)

        try:
            n, m = pileup_matrix.shape
        except ValueError:
            print(pileup_matrix)
            return None

        encoding = numpy.zeros([1, m], dtype=numpy.float64)

        # print("BEGIN")
        for column_index in range(m):
            column = pileup_matrix[:, column_index]
            unique, inverse = numpy.unique(column, return_inverse=True)
            bincount = numpy.bincount(inverse)
            mode = bincount.argmax()
            consensus_code = unique[mode]

            # print("----------")
            # print("index\t\t", column_index)
            # print("unique'\t\t", unique)
            # print("bincount\t", bincount)
            # print("max\t\t", mode)
            # print("consensus\t", consensus_code)

            encoding[0, column_index] = round(consensus_code, 3)

        return encoding

    def get_normalized_frequencies(self, pileup_matrix):
        """
        Create a binned normalized count of each column, with dimensions [5,c] where c is the number of columns in
        pileup, and 5 is the number of symbols allowed: [-,A,G,T,C]
        :param pileup_matrix:
        :param format:
        :return:
        """
        pileup_matrix = pileup_matrix.round(3)
        pileup_matrix = numpy.atleast_2d(pileup_matrix)

        n, m = pileup_matrix.shape

        one_hot = numpy.zeros([5, m])

        for column_index in range(m):
            column = pileup_matrix[:, column_index]
            unique, inverse = numpy.unique(column, return_inverse=True)
            bincount = numpy.bincount(inverse)

            n_characters = unique.shape[0]
            for i in range(n_characters):
                consensus_code = unique[i]
                character_index = self.float_to_index[consensus_code]
                frequency = bincount[i]

                one_hot[character_index, column_index] = frequency

        # normalize
        column_sums = numpy.sum(one_hot, axis=0)
        one_hot = one_hot/column_sums

        return one_hot

    def get_avg_repeat_counts(self, pileup_matrix, repeat_matrix):
        """
        For a pileup and repeat matrix return a 2 channel matrix with the normalized frequency and average repeat count
        for each allele in the pileup.
        :param pileup_matrix:
        :param repeat_matrix:
        :return:
        """

        pileup_matrix = pileup_matrix.round(3)

        n, m = pileup_matrix.shape

        repeat_counts = numpy.zeros([5, m])

        for column_index in range(m):
            pileup_column = pileup_matrix[:, column_index]
            repeat_column = repeat_matrix[:, column_index]

            for c,character in enumerate(["-","A","G","T","C"]):
                float_value = sequence_to_float[character]

                mask = (pileup_column == float_value)

                repeats = repeat_column[mask]

                n = int(numpy.count_nonzero(mask))

                if n == 0:
                    average = 0
                else:
                    total = float(numpy.sum(repeats))
                    average = total/n

                repeat_counts[c,column_index] = average

        repeat_counts = numpy.round(repeat_counts,3)

        return repeat_counts

    def get_repeat_counts(self, pileup_matrix, repeat_matrix):
        """
        For a pileup and repeat matrix return a 2 channel matrix with the normalized frequency and average repeat count
        for each allele in the pileup.
        :param pileup_matrix:
        :param repeat_matrix:
        :return:
        """
        pileup_matrix = pileup_matrix.round(3)

        n, m = pileup_matrix.shape

        repeat_counts = numpy.zeros([5, m])

        for column_index in range(m):
            pileup_column = pileup_matrix[:, column_index]
            repeat_column = repeat_matrix[:, column_index]

            for c,character in enumerate(["-","A","G","T","C"]):
                float_value = sequence_to_float[character]

                mask = (pileup_column == float_value)

                repeats = repeat_column[mask]

                n = int(numpy.count_nonzero(mask))

                if n == 0:
                    average = 0
                else:
                    total = float(numpy.sum(repeats))
                    average = total/n

                repeat_counts[c,column_index] = average

        repeat_counts = numpy.round(repeat_counts,3)

        return repeat_counts

    def mode(self, x):
        unique, inverse = numpy.unique(x, return_inverse=True)
        bincount = numpy.bincount(inverse)
        mode_index = bincount.argmax()
        mode = unique[mode_index]

        # print()
        # print(x)
        # print(unique)
        # print(inverse)
        # print(bincount)
        # print(mode)

        return mode

    def call_repeat_consensus_as_integer_vector(self, repeat_matrix, pileup_matrix, consensus_encoding):
        """
        For a repeat matrix which encodes the number of repeats for each character in a pileup of aligned sequences,
        determine the consensus number of repeats for each column.
        :param repeat_matrix:
        :return:
        """
        pileup_matrix = pileup_matrix.round(3)
        pileup_matrix = numpy.atleast_2d(pileup_matrix)

        consensus_encoding = consensus_encoding.round(3)

        n, m = pileup_matrix.shape

        repeat_consensus = numpy.zeros([1, m])

        for column_index in range(m):
            pileup_column = pileup_matrix[:, column_index]
            repeat_column = repeat_matrix[:, column_index]
            pileup_column_consensus = consensus_encoding[0, column_index]

            mask = (pileup_column == pileup_column_consensus)

            # print(repeat_column.shape)
            # print(mask.shape)

            repeats = repeat_column[mask]

            # exit()

            column_repeat_consensus = self.mode(repeats)
            repeat_consensus[0,column_index] = column_repeat_consensus

        repeat_consensus = numpy.round(repeat_consensus)

        return repeat_consensus

    def get_consensus_repeats(self, repeat_matrix, pileup_matrix, consensus_encoding):
        """
        For a repeat matrix which encodes the number of repeats for each character in a pileup of aligned sequences,
        retrieve the repeats that correspond to the consensus character at each column.
        :param repeat_matrix:
        :return:
        """
        pileup_matrix = pileup_matrix.round(3)
        pileup_matrix = numpy.atleast_2d(pileup_matrix)

        consensus_encoding = consensus_encoding.round(3)

        n, m = pileup_matrix.shape

        repeats = list()

        for column_index in range(m):
            pileup_column = pileup_matrix[:, column_index]
            repeat_column = repeat_matrix[:, column_index]
            pileup_column_consensus = consensus_encoding[0, column_index]

            mask = (pileup_column == pileup_column_consensus)

            column_repeats = repeat_column[mask]

            repeats.append(column_repeats)

        return repeats

    def expand_collapsed_consensus_as_one_hot(self, consensus_encoding, repeat_consensus_encoding):
        # blanks coded as 0 repeats, but for comparison sake, we want to include blanks
        n_spaces = repeat_consensus_encoding.shape[1] - numpy.count_nonzero(repeat_consensus_encoding)

        # print(n_spaces)

        run_length = int(numpy.sum(repeat_consensus_encoding, axis=1)) + n_spaces
        expanded_one_hot = numpy.zeros([5, run_length])

        n, m = repeat_consensus_encoding.shape

        expanded_index = 0
        for collapsed_index in range(m):
            # blanks coded as 0 repeats, but for comparison sake, we want to include blanks
            n_repeats = max(1,int(repeat_consensus_encoding[:,collapsed_index]))
            consensus_code = float(consensus_encoding[:,collapsed_index])

            character_index = self.float_to_index[consensus_code]

            # print(self.index_to_sequence[character_index], character_index, repeat_consensus_encoding[:,collapsed_index])

            for i in range(n_repeats):
                expanded_one_hot[character_index, expanded_index] = 1
                expanded_index += 1

        return expanded_one_hot

    def expand_collapsed_consensus_as_string(self, consensus_encoding, repeat_consensus_encoding, ignore_spaces=False):
        # blanks coded as 0 repeats, but for comparison sake, we ant to include blanks
        n_spaces = repeat_consensus_encoding.shape[1] - numpy.count_nonzero(repeat_consensus_encoding)

        consensus_characters = list()

        n, m = repeat_consensus_encoding.shape

        expanded_index = 0
        for collapsed_index in range(m):
            # blanks coded as 0 repeats, but for comparison sake, we ant to include blanks
            n_repeats = max(1,int(repeat_consensus_encoding[:,collapsed_index]))
            consensus_code = float(consensus_encoding[:,collapsed_index])

            character_index = self.float_to_index[consensus_code]

            if ignore_spaces:
                if self.index_to_sequence[character_index] == "-":
                    continue

            for i in range(n_repeats):
                consensus_characters.append(self.index_to_sequence[character_index])
                expanded_index += 1

        consensus_string = ''.join(consensus_characters)

        return consensus_string

    def decode_one_hot_to_string(self, one_hot_encoding, ignore_blanks=True):
        if type(one_hot_encoding) != numpy.ndarray:
            exit("Incompatible data type: " + str(type(one_hot_encoding)))

        consensus_characters = list()

        n, m = one_hot_encoding.shape

        indexes = numpy.argmax(one_hot_encoding, axis=0)

        for i in range(m):
            character_index = indexes[i]

            if ignore_blanks:
                if character_index != 0:
                    consensus_characters.append(self.index_to_sequence[character_index])
            else:
                consensus_characters.append(self.index_to_sequence[character_index])

        consensus_string = ''.join(consensus_characters)

        return consensus_string

    def encode_one_hot_as_float(self, one_hot_encoding):
        n, m = one_hot_encoding.shape

        encoding = numpy.zeros([1,m])

        indexes = numpy.argmax(one_hot_encoding, axis=0)

        for i in range(m):
            # blanks coded as 0 repeats, but for comparison sake, we ant to include blanks
            character_index = indexes[i]
            encoding[0,i] = sequence_to_float[self.index_to_sequence[character_index]]

        # consensus_string = ''.join(consensus_characters)
        # print(one_hot_encoding)
        # print(encoding)

        return encoding


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

    consensus_caller = ConsensusCaller(sequence_to_index, sequence_to_float)

    consensus_string = consensus_caller.call_consensus_as_string(pileup_matrix)
    # consensus_one_hot = consensus_caller.call_consensus_as_one_hot(pileup_matrix)
    consensus_normalized_frequency = consensus_caller.get_normalized_frequencies(pileup_matrix)
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


def test_collapsed_consensus_caller():
    alignments = [["0", "TG-TGC---T-GCTATC-G--C-G-T"],
                  ["1", "TG-TGC--AT-GC--T-----C-G-T"],
                  ["2", "TG-TGC--AT-GC--TC-G--T-G-T"],
                  ["3", "TG-TGC--AT-G---T-------G-T"],
                  ["4", "TG-TGC--AT-GC--T-----C-G-T"],
                  ["5", "TG-TGC--AT-GC--T--G--C---T"],
                  ["6", "TG-TGCT-AT-GC--T-----C-GAT"],
                  ["7", "TG-TGC--AT-GC--TC-T--C-G-T"],
                  ["8", "TG-TGC--AT-GC--TC-GTAT-G-T"],
                  ["9", "TG-TGC--AT-G---T-------G-T"],
                  ["10", "TG-TGC--AT-GC--T-----C-G-T"],
                  ["11", "TG-TGC--AT-GC--TC-T--C-G-T"],
                  ["12", "TG-TGCTGATAGC--T-------G-T"],
                  ["13", "TG-TGC--AT-G---T-----C-G-T"],
                  ["14", "TG-TGC--AT-GC--T-------GAT"],
                  ["15", "TGCTGC--AT-GC--TCTG--C----"],
                  ["16", "TG-TGC--ATAGC--T-------G-T"],
                  ["17", "TGCTGC--AT-GC--T-----C-G-T"],
                  ["18", "TG-TGC--AT-GC--T-------GAT"],
                  ["19", "TG-TGC--ATAGC--T-----C-G-T"],
                  ["20", "TG-TGC--AT-GC--T-----CAG-T"],
                  ["21", "TG-TGC--AT-GC--TC-G--T--A-"]]

    ref_alignment = [["ref", "TG-TGC--AT-GC--T-----C-G-T"]]
    expanded_ref_alignment = [["ref", "TTTG--T--TGCC-----ATT-GC--T-T-----TC--G-G--T"]]

    pileup_repeats = [[3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
                      [3, 1, 2, 1, 2, 1, 2, 1, 1, 4, 1, 1, 1],
                      [3, 1, 2, 1, 2, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1],
                      [2, 2, 2, 1, 2, 1, 1, 1, 4, 2, 1],
                      [2, 2, 2, 1, 2, 1, 2, 1, 1, 3, 1, 1, 1],
                      [3, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1],
                      [3, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1],
                      [3, 1, 2, 1, 2, 1, 2, 1, 1, 3, 1, 1, 1, 2, 1],
                      [3, 1, 2, 1, 2, 1, 2, 1, 1, 3, 1, 2, 1, 1, 1, 1, 1],
                      [3, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1],
                      [3, 1, 2, 1, 2, 1, 2, 1, 1, 3, 1, 1, 1],
                      [3, 1, 2, 1, 2, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1],
                      [3, 1, 2, 1, 2, 1, 1, 3, 2, 1, 1, 1, 3, 2, 1],
                      [3, 1, 2, 1, 2, 1, 2, 1, 4, 1, 3, 1],
                      [6, 1, 1, 1, 2, 1, 2, 1, 1, 3, 1, 1, 1],
                      [2, 1, 1, 3, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1],
                      [3, 1, 2, 1, 2, 1, 2, 1, 1, 1, 3, 2, 1],
                      [3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 4, 1, 1, 1],
                      [3, 1, 2, 1, 2, 2, 2, 1, 1, 3, 1, 1, 2],
                      [3, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2, 1],
                      [3, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1],
                      [3, 1, 2, 1, 2, 1, 2, 1, 1, 3, 1, 1, 1, 1]]

    ref_repeats = [[3, 1, 2, 1, 2, 1, 2, 1, 1, 3, 1, 2, 1]]

    # reference_matrix = convert_aligned_reference_to_one_hot(ref_alignment)
    pileup_matrix, pileup_repeat_matrix = convert_collapsed_alignments_to_matrix(alignments, pileup_repeats, fixed_coverage=False)
    reference_matrix, reference_repeat_matrix = convert_collapsed_alignments_to_matrix(ref_alignment, ref_repeats, fixed_coverage=False)

    # plot_collapsed_encodings(pileup_matrix=pileup_matrix,
    #                          reference_matrix=reference_matrix,
    #                          pileup_repeat_matrix=pileup_repeat_matrix,
    #                          reference_repeat_matrix=reference_repeat_matrix)

    consensus_caller = ConsensusCaller(sequence_to_index, sequence_to_float)

    pileup_consensus = consensus_caller.call_consensus_as_encoding(pileup_matrix)
    repeat_consensus = consensus_caller.call_repeat_consensus_as_integer_vector(repeat_matrix=pileup_repeat_matrix,
                                                                                pileup_matrix=pileup_matrix,
                                                                                consensus_encoding=pileup_consensus)

    consensus_expanded_reference_matrix = consensus_caller.expand_collapsed_consensus_as_one_hot(consensus_encoding=pileup_consensus,
                                                                                                 repeat_consensus_encoding=repeat_consensus)

    expanded_reference_matrix = consensus_caller.expand_collapsed_consensus_as_one_hot(consensus_encoding=reference_matrix,
                                                                                       repeat_consensus_encoding=reference_repeat_matrix)

    consensus_reference_string = consensus_caller.expand_collapsed_consensus_as_string(consensus_encoding=pileup_consensus,
                                                                                       repeat_consensus_encoding=repeat_consensus)

    reference_string = consensus_caller.expand_collapsed_consensus_as_string(consensus_encoding=reference_matrix,
                                                                             repeat_consensus_encoding=reference_repeat_matrix)

    # print(reference_repeat_matrix)
    # print(repeat_consensus.round())

    # print(reference_matrix)
    # print(consensus_expanded_reference_matrix)

    # print(consensus_reference_string)
    # print(reference_string)

    character_frequencies = consensus_caller.get_normalized_frequencies(pileup_matrix=pileup_matrix)
    repeat_counts = consensus_caller.get_avg_repeat_counts(pileup_matrix=pileup_matrix, repeat_matrix=pileup_repeat_matrix)

    pyplot.imshow(character_frequencies)
    pyplot.show()
    pyplot.imshow(repeat_counts)
    pyplot.show()

    # fig, axes = pyplot.subplots(nrows=2)
    # axes[0].imshow(reference_repeat_matrix)
    # axes[1].imshow(repeat_consensus)
    #
    # pyplot.show()
    # pyplot.close()
    #
    # fig, axes = pyplot.subplots(nrows=2)
    # axes[0].imshow(consensus_expanded_reference_matrix)
    # axes[1].imshow(expanded_reference_matrix)
    #
    # pyplot.show()


if __name__ == "__main__":
    # test_consensus_caller()
    test_collapsed_consensus_caller()
