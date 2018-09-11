from subprocess import Popen, PIPE


GLOBAL_ALIGN = True
MATCH_SCORE = 1
MISMATCH_SCORE = -2
GAP_SCORE = -1


def call_commandline_spoa(space_separated_sequences, ref_sequence, two_pass=True):
    match_arg = str(MATCH_SCORE)
    mismatch_arg = str(MISMATCH_SCORE)
    gap_arg = str(GAP_SCORE)
    alignment_arg = str(1) if GLOBAL_ALIGN else str(0)

    if two_pass:
        script_path = "/home/ryan/software/spoa/align_sequences_and_reference_two_pass"
    else:
        script_path = "/home/ryan/software/spoa/align_sequences_and_reference"

    args = [script_path,
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


def get_spoa_alignment(sequences, ref_sequence, two_pass=True):
    space_separated_sequences = ' '.join(sequences)

    read_alignment_strings, ref_alignment_string = call_commandline_spoa(space_separated_sequences, ref_sequence, two_pass=two_pass)

    alignments = list()
    for a, alignment_string in enumerate(read_alignment_strings):
        label = str(a)
        alignment = [label, alignment_string]

        alignments.append(alignment)

    ref_label = "ref"
    ref_alignment = [ref_label, ref_alignment_string]

    return alignments, ref_alignment
