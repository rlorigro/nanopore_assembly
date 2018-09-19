from modules.pileup_utils import *
from modules.alignment_utils import *
from modules.ConsensusCaller import ConsensusCaller
from modules.train_test_utils import plot_runlength_prediction


def parse_repeat_strings(repeat_strings):
    repeats = list()
    for repeat_string in repeat_strings:
        repeat_list = list()
        for character in repeat_string:
            repeat_list.append(int(character))

        repeats.append(repeat_list)

    return repeats


def assemble_from_strings(print_results=False, plot_results=False):
    consensus_caller = ConsensusCaller(sequence_to_float=sequence_to_float, sequence_to_index=sequence_to_index)

    sequences = ["ATCGTGCAGACGTCTGCAGTGCTATCACTGCATCTGTCAGCTGCGCGACACGATCGCATCTCACTCTGCA",
                 "GACACGTGTGACGTCGCAGTGCTGCATCACGTGCATCGTCGAGCTGCGCGACATCGCATCTCACTCTGCA",
                 "GATATCGTGCAGACGTCGCAGTGCTATCTCTACGTAGACATCGTCAGACGCTGCGCGACATCGATCGCATCTCACTCTGCA",
                 "GATATCGTGCAGACGTCTGCAGTGCTATCACGTATCTCGACGCTGCGCGACATCGATCGCATCTCACTCTGCA",
                 "GATATCGTGCAGACGTCTGCAGTGCTATCACTGTATCGTCGACGCTGCGCGACATCGATCGCATCTCACTCGCA",
                 "GATATCGTGCAGACGTCTGCAGTGCTATCAGCGTATCGTCGACGCTGCGCGACATCGATCGCATCTCACTCTGCA",
                 "GATATCGTGCAGACGTCTGCTCAGTGCTATCACAGCATCGTCGAGCGCTGCGCGACACGATCGCATCTCACTCTGCA",
                 "GATATCGTGCGACGTCTGCAGTGCTATCACAGTATCGTCGACGCTGCGCGACATCGATCGCATCTCACTCTGCA",
                 "GATGATATGTGCAGACGTCTGCAGTGCTATCACAGTATCGTCGACGCTGCGCGACATCGATCGCATCTCACTCTATATCGCGCA"]

    repeat_strings = ["2111221111115111411131121421111112121122211111112122111221111121312111",
                      "1111212211111112411131113215221111213111122111111112111221111121312211",
                      "111211122111111511411131121111122111111112111111111111111121112111221111121312211",
                      "1112111221111115111411131121523211151112111111111121112111221111121312211",
                      "11121112211111151114111311213211111131111211111111112111211122111112131211",
                      "111211122111111511141113112151122111311112111111111121112111221111121312211",
                      "11121112211111151111131113112222211111311111111111111112112111221111121312111",
                      "11121112211111511141113112142211111311112111111111121112111221111121312211",
                      "111111121221111114111411131121522111113111121111111111211121112211111213111121111211"]

    repeats = parse_repeat_strings(repeat_strings)
    alignments = get_spoa_alignment_no_ref(sequences=sequences)

    x_pileup, x_repeat = convert_collapsed_alignments_to_matrix(alignments, repeats)

    # remove padding
    x_pileup = trim_empty_rows(x_pileup, background_value=sequence_to_float["-"])
    x_repeat = trim_empty_rows(x_repeat, background_value=sequence_to_float["-"])

    # use consensus caller on bases and repeats independently
    y_pileup_predict = consensus_caller.call_consensus_as_encoding(x_pileup)
    y_repeat_predict = consensus_caller.call_repeat_consensus_as_integer_vector(repeat_matrix=x_repeat,
                                                                                pileup_matrix=x_pileup,
                                                                                consensus_encoding=y_pileup_predict)

    # decode as string to compare with non-runlength version
    expanded_consensus_string = \
        consensus_caller.expand_collapsed_consensus_as_string(consensus_encoding=y_pileup_predict,
                                                              repeat_consensus_encoding=y_repeat_predict,
                                                              ignore_spaces=True)

    if plot_results:
        x_repeat = numpy.array(x_repeat)
        plot_runlength_prediction(x_pileup=x_pileup,
                                  y_pileup=y_pileup_predict,
                                  x_repeat=x_repeat,
                                  y_repeat=y_repeat_predict,
                                  title=expanded_consensus_string)

    if print_results:
        for label, alignstring in alignments:
            print("{0:15s} {1:s}".format(label, alignstring))

        # print(repeats)


if __name__ == "__main__":
    assemble_from_strings(plot_results=True, print_results=True)