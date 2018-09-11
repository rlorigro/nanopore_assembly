from os import walk, path
from matplotlib import pyplot
import numpy
import torch


def plot_collapsed_encodings(pileup_matrix, pileup_repeat_matrix, reference_matrix, reference_repeat_matrix):
    print(pileup_matrix.shape, pileup_repeat_matrix.shape, reference_matrix.shape, reference_repeat_matrix.shape)

    ratio = pileup_matrix.shape[0] / reference_matrix.shape[0]
    fig, axes = pyplot.subplots(nrows=4, gridspec_kw={'height_ratios': [1, ratio, 1, ratio]})

    pileup_matrix = pileup_matrix.squeeze()
    pileup_repeat_matrix = pileup_repeat_matrix.squeeze()
    # reference_matrix = reference_matrix.squeeze()
    # reference_repeat_matrix = reference_repeat_matrix.squeeze()

    axes[3].imshow(pileup_repeat_matrix)
    axes[2].imshow(reference_repeat_matrix)
    axes[1].imshow(pileup_matrix)
    axes[0].imshow(reference_matrix)

    axes[3].set_ylabel("repeats")
    axes[2].set_ylabel("ref repeats")
    axes[1].set_ylabel("nucleotide")
    axes[0].set_ylabel("reference")

    pyplot.show()
    pyplot.close()


def get_all_file_paths_by_type(parent_directory_path, file_extension, sort=True):
    """
    Given a parent directory, iterate all files within, and return those that end in the extension provided by user.
    File paths returned in sorted order by default.
    :param parent_directory_path:
    :param file_extension:
    :param sort:
    :return:
    """
    all_files = list()

    for root, dirs, files in walk(parent_directory_path):
        sub_files = [path.join(root,subfile) for subfile in files if subfile.endswith(file_extension)]
        all_files.extend(sub_files)

    if sort:
        all_files.sort()

    return all_files


class DataLoader:
    def __init__(self, file_paths, batch_size, parse_batches=True, use_gpu=False):
        self.file_paths = file_paths

        self.path_iterator = iter(file_paths)
        self.n_files = len(file_paths)
        self.files_loaded = 0

        self.use_gpu = use_gpu

        self.batch_size = batch_size
        self.parse_batches = parse_batches

    def load_next_file(self, path_cache, x_pileup_cache, y_pileup_cache, x_repeat_cache, y_repeat_cache):
        """
        Assuming there is another file in the list of paths, load it and concatenate with the leftover entries from last
        :return:
        """
        next_path = next(self.path_iterator)

        x_pileup = numpy.load(next_path)['a']
        y_pileup = numpy.load(next_path)['b']
        x_repeat = numpy.load(next_path)['c']
        y_repeat = numpy.load(next_path)['d']

        # add 3rd dimension
        x_pileup = numpy.expand_dims(x_pileup, axis=0)
        y_pileup = numpy.expand_dims(y_pileup, axis=0)
        x_repeat = numpy.expand_dims(x_repeat, axis=0)
        y_repeat = numpy.expand_dims(y_repeat, axis=0)

        x_pileup_cache.append(x_pileup)
        y_pileup_cache.append(y_pileup)
        x_repeat_cache.append(x_repeat)
        y_repeat_cache.append(y_repeat)

        path_cache.append(next_path)

        self.files_loaded += 1

        return path_cache, x_pileup_cache, y_pileup_cache, x_repeat_cache, y_repeat_cache

    def load_batch(self):
        path_cache = list()
        x_pileup_cache = list()
        y_pileup_cache = list()
        x_repeat_cache = list()
        y_repeat_cache = list()

        while len(x_pileup_cache) < self.batch_size and self.files_loaded < self.n_files:
            path_cache, x_pileup_cache, y_pileup_cache, x_repeat_cache, y_repeat_cache = \
                self.load_next_file(path_cache, x_pileup_cache, y_pileup_cache, x_repeat_cache, y_repeat_cache)

        return path_cache, x_pileup_cache, y_pileup_cache, x_repeat_cache, y_repeat_cache

    def parse_batch(self, x_pileup_batch, y_pileup_batch, x_repeat_batch, y_repeat_batch):
        x_dtype = torch.FloatTensor
        # y_dtype = torch.FloatTensor  # for MSE Loss or BCE loss
        y_dtype = torch.LongTensor      # for CE Loss

        x_pileup_batch = torch.from_numpy(x_pileup_batch).type(x_dtype)
        y_pileup_batch = torch.from_numpy(y_pileup_batch).type(y_dtype)
        x_repeat_batch = torch.from_numpy(x_repeat_batch).type(x_dtype)
        y_repeat_batch = torch.from_numpy(y_repeat_batch).type(y_dtype)

        return x_pileup_batch, y_pileup_batch, x_repeat_batch, y_repeat_batch

    def __next__(self):
        """
        Get the next batch data. DOES NOT RETURN FINAL BATCH IF != BATCH SIZE
        :return:
        """
        path_cache, x_pileup_cache, y_pileup_cache, x_repeat_cache, y_repeat_cache = self.load_batch()

        if len(path_cache) > 0:
            x_pileup_batch = numpy.concatenate(x_pileup_cache, axis=0)
            y_pileup_batch = numpy.concatenate(y_pileup_cache, axis=0)
            x_repeat_batch = numpy.concatenate(x_repeat_cache, axis=0)
            y_repeat_batch = numpy.concatenate(y_repeat_cache, axis=0)

            assert x_pileup_batch.shape[0] == self.batch_size
            assert y_pileup_batch.shape[0] == self.batch_size
            assert x_repeat_batch.shape[0] == self.batch_size
            assert y_repeat_batch.shape[0] == self.batch_size

            if self.parse_batches:
                x_pileup_batch, y_pileup_batch, x_repeat_batch, y_repeat_batch = \
                    self.parse_batch(x_pileup_batch, y_pileup_batch, x_repeat_batch, y_repeat_batch)
        else:
            self.__init__(file_paths=self.file_paths,
                          batch_size=self.batch_size,
                          parse_batches=self.parse_batches,
                          use_gpu=self.use_gpu)

            raise StopIteration

        return path_cache, x_pileup_batch, y_pileup_batch, x_repeat_batch, y_repeat_batch

    def __iter__(self):
        self.load_batch()

        return self


if __name__ == "__main__":
    directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-9-4-12-26-1-1-247"
    file_paths = get_all_file_paths_by_type(parent_directory_path=directory, file_extension=".npz")

    data_loader = DataLoader(file_paths=file_paths, batch_size=1, parse_batches=False)

    for path_cache, x_pileup, y_pileup, x_repeat, y_repeat in data_loader:
        print(x_pileup.shape, y_pileup.shape, x_repeat.shape, y_repeat.shape)

        plot_collapsed_encodings(pileup_matrix=x_pileup[0,:,:],
                                 reference_matrix=y_pileup[0,:,:],
                                 pileup_repeat_matrix=x_repeat[0,:,:],
                                 reference_repeat_matrix=y_repeat[0,:,:])
