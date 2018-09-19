from modules.pileup_utils import sequence_to_float, sequence_to_index, visualize_matrix, trim_empty_rows
from modules.ConsensusCaller import ConsensusCaller
import numpy
import torch
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))


# def visualize_matrix(matrix):
#     pyplot.imshow(matrix, cmap="viridis")
#     pyplot.show()


# def get_all_file_paths_by_type(parent_directory_path, file_extension, sort=True):
#     """
#     Given a parent directory, iterate all files within, and return those that end in the extension provided by user.
#     File paths returned in sorted order by default.
#     :param parent_directory_path:
#     :param file_extension:
#     :param sort:
#     :return:
#     """
#     all_files = list()
#
#     for root, dirs, files in walk(parent_directory_path):
#         sub_files = [path.join(root,subfile) for subfile in files if subfile.endswith(file_extension)]
#         all_files.extend(sub_files)
#
#     if sort:
#         all_files.sort()
#
#     return all_files


class DataLoader:
    def __init__(self, file_paths, batch_size, parse_batches=True, use_gpu=False, convert_to_frequency=False):
        self.file_paths = file_paths

        self.path_iterator = iter(file_paths)
        self.n_files = len(file_paths)
        self.files_loaded = 0

        self.use_gpu = use_gpu

        self.batch_size = batch_size
        self.parse_batches = parse_batches
        self.convert_to_frequency = convert_to_frequency

    def __len__(self):
        return self.n_files

    def load_next_file(self, path_cache, x_cache, y_cache):
        """
        Assuming there is another file in the list of paths, load it and concatenate with the leftover entries from last
        :return:
        """
        next_path = next(self.path_iterator)

        x = numpy.load(next_path)['a']
        y = numpy.load(next_path)['b']

        # add 3rd dimension
        x = numpy.expand_dims(x, axis=0)
        y = numpy.expand_dims(y, axis=0)

        x_cache.append(x)
        y_cache.append(y)
        path_cache.append(next_path)

        self.files_loaded += 1

        return path_cache, x_cache, y_cache

    def load_batch(self):
        path_cache = list()
        x_cache = list()
        y_cache = list()

        while len(x_cache) < self.batch_size and self.files_loaded < self.n_files:
            path_cache, x_cache, y_cache = self.load_next_file(path_cache, x_cache, y_cache)

        return path_cache, x_cache, y_cache

    def parse_batch(self, x_batch, y_batch):
        x_dtype = torch.FloatTensor
        # y_dtype = torch.FloatTensor  # for MSE Loss or BCE loss
        y_dtype = torch.LongTensor      # for CE Loss

        x = torch.from_numpy(x_batch).type(x_dtype)
        y = torch.from_numpy(y_batch).type(y_dtype)

        return x, y

    def convert_pileup_to_frequency(self, x_batch):
        caller = ConsensusCaller(sequence_to_float=sequence_to_float, sequence_to_index=sequence_to_index)

        batch_size, height, width = x_batch.shape
        # print("before", x_batch.shape)

        frequency_matrices = list()
        for b in range(batch_size):
            x_pileup = x_batch[b,:,:]

            x_pileup = trim_empty_rows(x_pileup, background_value=sequence_to_float["-"])

            # print(type(x_pileup))
            # print(x_pileup.shape)
            normalized_frequencies = caller.get_normalized_frequencies(x_pileup)
            normalized_frequencies = numpy.expand_dims(normalized_frequencies, axis=0)

            frequency_matrices.append(normalized_frequencies)

        x_batch = numpy.concatenate(frequency_matrices, axis=0)

        # print("after", x_batch.shape)

        return x_batch

    def __next__(self):
        """
        Get the next batch data. DOES NOT RETURN FINAL BATCH IF != BATCH SIZE
        :return:
        """
        path_cache, x_cache, y_cache = self.load_batch()

        if len(path_cache) > 0:
            x_batch = numpy.concatenate(x_cache, axis=0)
            y_batch = numpy.concatenate(y_cache, axis=0)

            assert x_batch.shape[0] == self.batch_size
            assert y_batch.shape[0] == self.batch_size

            if self.convert_to_frequency:
                x_batch = self.convert_pileup_to_frequency(x_batch)

            if self.parse_batches:
                x_batch, y_batch = self.parse_batch(x_batch, y_batch)

        else:
            self.__init__(file_paths=self.file_paths,
                          batch_size=self.batch_size,
                          parse_batches=self.parse_batches,
                          use_gpu=self.use_gpu)

            raise StopIteration

        return path_cache, x_batch, y_batch

    def __iter__(self):
        self.load_batch()

        return self


if __name__ == "__main__":
    from handlers.FileManager import FileManager

    directory = "output/run_2018-8-22-15-10-37-2-234"
    file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=directory, file_extension=".npz")

    data_loader = DataLoader(file_paths=file_paths, batch_size=1, parse_batches=False)

    for x,y in data_loader:
        print(x.shape, y.shape)

        for i in range(x.shape[0]):
            visualize_matrix(x[i,:])
            visualize_matrix(y[i,:])
