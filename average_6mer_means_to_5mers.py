from collections import defaultdict


def main():
    path = "/home/ryan/data/Nanopore/kmerMeans"
    path = "/home/ryan/data/Nanopore/r9.4_180mv_450bps_6mer_template_median68pA.model"

    kmer_means = defaultdict(list)

    with open(path, 'r') as file:
        for l,line in enumerate(file):
            if l > 0:   # ignore header
                line = line.strip().split("\t")
                kmer = line[0]
                mean = float(line[1])

                five_mer = kmer[:-1]
                kmer_means[five_mer].append(mean)

                # print(kmer, five_mer, mean)

    average_kmer_means = dict()

    for kmer in kmer_means:
        means = kmer_means[kmer]
        length = len(means)
        mean_mean = sum(means)/length
        average_kmer_means[kmer] = mean_mean

        if length != 4:
            exit("ERROR, less than 4 6-mers found for 5-mer: "+kmer)

        print(kmer, means, mean_mean)


if __name__ == "__main__":
    main()
