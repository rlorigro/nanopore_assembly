

class FastaWriter:
    def __init__(self, output_file_path):
        self.file_path = output_file_path

    def write_sequences(self, sequences, labels=None):
        with open(self.file_path, 'w') as file:
            for i in range(len(sequences)):
                string = self.generate_sequence_string(sequences, labels, i)

                file.write(string)

    def generate_label_string(self, labels, i):
        if labels is None:
            label = str(i)
        else:
            label = labels[i]

        label_string = ">" + label

        return label_string

    def generate_sequence_string(self, sequences, labels, i):
        label = self.generate_label_string(labels, i)
        sequence = sequences[i]

        if i == 0:
            entry = [label,sequence]
        else:
            entry = ['',label,sequence]

        string = '\n'.join(entry)

        return string


def test_fasta_writer():
    path = "output/fasta_test.txt"
    sequences = ["ACTG","CTGA","TGAC","GACT"]

    writer = FastaWriter(output_file_path=path)

    writer.write_sequences(sequences)


if __name__ == "__main__":
    test_fasta_writer()