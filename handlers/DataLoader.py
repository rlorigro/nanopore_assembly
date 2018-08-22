from os import walk, path
from matplotlib import pyplot
import numpy
import torch


def visualize_matrix(matrix):
    pyplot.imshow(matrix, cmap="viridis")
    pyplot.show()


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

    def load_next_file(self, x_cache, y_cache):
        """
        Assuming there is another file in the list of paths, load it and concatenate with the leftover entries from last
        :return:
        """
        next_path = next(self.path_iterator)

        x = numpy.load(next_path)['a']
        y = numpy.load(next_path)['b']

        print(x.shape)
        print(y.shape)

        # add 3rd dimension
        x = numpy.expand_dims(x, axis=0)
        y = numpy.expand_dims(y, axis=0)

        x_cache.append(x)
        y_cache.append(y)

        self.files_loaded += 1

        return x_cache, y_cache

    def load_batch(self):
        x_cache = list()
        y_cache = list()

        while len(x_cache) < self.batch_size:
            if self.files_loaded < self.n_files:
                x_cache, y_cache = self.load_next_file(x_cache, y_cache)
            else:
                raise StopIteration

        return x_cache, y_cache

    def parse_batch(self, x_batch, y_batch):
        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor  # for MSE Loss or BCE loss
        # y_dtype = torch.LongTensor      # for CE Loss

        x = torch.from_numpy(x_batch).type(x_dtype)
        y = torch.from_numpy(y_batch).type(y_dtype)

        return x, y

    def __next__(self):
        """
        Get the next batch data. DOES NOT RETURN FINAL BATCH IF != BATCH SIZE
        :return:
        """
        x_cache, y_cache = self.load_batch()

        x_batch = numpy.concatenate(x_cache, axis=0)
        y_batch = numpy.concatenate(y_cache, axis=0)

        print(x_batch.shape, y_batch.shape)

        assert x_batch.shape[0] == self.batch_size
        assert y_batch.shape[0] == self.batch_size

        if self.parse_batches:
            x_batch, y_batch = self.parse_batch(x_batch, y_batch)

        return x_batch, y_batch

    def __iter__(self):
        self.load_batch()

        return self


if __name__ == "__main__":
    directory = "output/run_2018-8-21-17-8-37-1-233"
    file_paths = get_all_file_paths_by_type(parent_directory_path=directory, file_extension=".npz")

    data_loader = DataLoader(file_paths=file_paths, batch_size=1, parse_batches=False)

    for x,y in data_loader:
        print(x.shape, y.shape)

        for i in range(x.shape[0]):
            visualize_matrix(x[i,:])
            visualize_matrix(y[i,:])
