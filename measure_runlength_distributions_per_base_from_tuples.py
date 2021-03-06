# from modules.pileup_utils import sequence_to_float, sequence_to_index, trim_empty_rows, index_to_float, A, G, T, C
# from modules.ConsensusCaller import ConsensusCaller
# from handlers.DataLoaderRunlength import DataLoader
# from handlers.FileManager import FileManager
# from collections import defaultdict
# from matplotlib import pyplot
# import datetime
# import numpy
# import sys
# import os
#
# numpy.set_printoptions(linewidth=400, threshold=100000, suppress=True, precision=3)
#
#
# FREQUENCY_THRESHOLD = 0.7
#
#
# def save_numpy_matrices(output_dir, filename, matrices):
#     array_file_extension = ".npz"
#
#     # ensure chromosomal directory exists
#     if not os.path.exists(output_dir):
#         FileManager.ensure_directory_exists(output_dir)
#
#     output_path_prefix = os.path.join(output_dir, filename)
#
#     output_path = output_path_prefix + array_file_extension
#
#     # write numpy arrays
#     numpy.savez_compressed(output_path, a=matrices[0], g=matrices[1], t=matrices[2], c=matrices[3])
#
#
# def convert_indices_to_float_encodings(indices):
#     encoding = indices*0.2
#     encoding = encoding.reshape([1,encoding.shape[0]])
#
#     return encoding
#
#
# def plot_runlength_distributions(runlengths_per_base):
#     for base_runlengths in runlengths_per_base:
#
#         x_min = 0
#         x_max = max(base_runlengths.keys()) + 1
#
#         fig, axes = pyplot.subplots(nrows=len(base_runlengths), sharex=True, sharey=True)
#
#         for k,key in enumerate(sorted(base_runlengths)):
#             runlength_values = base_runlengths[key]
#
#             step = 1
#             bins = numpy.arange(x_min, x_max + step, step=step)
#             frequencies, bins = numpy.histogram(runlength_values, bins=bins, normed=True)
#
#             center = (bins[:-1] + bins[1:]) / 2 - step/2
#
#             axes[k].bar(center, frequencies, width=step, align="center")
#             axes[k].set_ylabel(str(key))
#             axes[k].set_xticks(numpy.arange(x_min, x_max+1))
#
#         axes[len(base_runlengths)-1].set_xlabel("Observed run length")
#         pyplot.show()
#         pyplot.close()
#
#
# def get_runlengths(x_pileup, x_repeat, y_pileup, y_repeat):
#     lengths_per_base = [defaultdict(list) for base in [A, G, T, C]]
#
#     consensus_caller = ConsensusCaller(sequence_to_float=sequence_to_float, sequence_to_index=sequence_to_index)
#
#     allele_frequencies = consensus_caller.get_normalized_frequencies(pileup_matrix=x_pileup)
#
#     x_pileup_consensus = numpy.argmax(allele_frequencies, axis=0)
#     x_pileup_max_encoding = convert_indices_to_float_encodings(x_pileup_consensus)
#
#     x_repeat = consensus_caller.get_consensus_repeats(repeat_matrix=x_repeat,
#                                                       pileup_matrix=x_pileup,
#                                                       consensus_encoding=x_pileup_max_encoding)
#
#     for i in range(len(y_repeat)):
#         y = int(y_repeat[:,i])
#         x = x_repeat[i]
#
#         base_index = x_pileup_consensus[i] - 1
#
#         # print('\n', base_index, x, y)
#
#         lengths_per_base[base_index][y].append(x)
#
#     return lengths_per_base
#
#
# def encode_runlength_distributions_as_matrix(runlengths_per_base, log_scale=False, normalize_observed=False):
#     base_frequency_matrices = list()
#
#     for b, base_runlengths in enumerate(runlengths_per_base):
#         x_min = 0
#         x_max = max(base_runlengths.keys()) + 1
#         length = x_max - x_min
#
#         frequencies_list = list()
#
#         for i in range(x_min,x_max):
#             if i in base_runlengths:
#                 runlength_values = base_runlengths[i]
#
#                 step = 1
#                 bins = numpy.arange(x_min, x_max + step, step=step)
#                 frequencies, bins1 = numpy.histogram(runlength_values, bins=bins, normed=normalize_observed)
#                 frequencies = frequencies.reshape([1,frequencies.shape[0]])
#             else:
#                 frequencies = numpy.zeros([1,length])
#
#             frequencies_list.append(frequencies)
#
#         frequencies = numpy.concatenate(frequencies_list, axis=0)
#
#         if log_scale:
#             frequencies = numpy.log10(frequencies + 1)
#
#         base_frequency_matrices.append(frequencies)
#
#     return base_frequency_matrices
#
#
# def measure_runlengths(data_loader):
#     all_runlengths_per_base = [defaultdict(list) for base in [A, G, T, C]]
#
#     n_files = len(data_loader)
#     n = 0
#
#     print("testing n windows: ", n_files)
#
#     for paths, x_pileup, y_pileup, x_repeat, y_repeat in data_loader:
#         x_pileup = trim_empty_rows(x_pileup[0,:,:], background_value=sequence_to_float["-"])
#         y_pileup = y_pileup[0,:,:]
#         x_repeat = trim_empty_rows(x_repeat[0,:,:], background_value=sequence_to_float["-"])
#         y_repeat = y_repeat[0,:,:]
#
#         x_pileup = numpy.atleast_2d(x_pileup)
#         y_pileup = numpy.atleast_2d(y_pileup)
#         x_repeat = numpy.atleast_2d(x_repeat)
#         y_repeat = numpy.atleast_2d(y_repeat)
#
#         # try:
#         runlengths_per_base = get_runlengths(x_pileup=x_pileup, x_repeat=x_repeat, y_pileup=y_pileup, y_repeat=y_repeat)
#         # except IndexError as e:
#         #     print()
#         #     print(e)
#         #     # pyplot.imshow(x_repeat)
#         #     # pyplot.show()
#         #     # pyplot.close()
#         #     # pyplot.imshow(x_pileup)
#         #     # pyplot.show()
#         #     # pyplot.close()
#         #     continue
#
#         for b,base_runlengths in enumerate(runlengths_per_base):
#             for key in base_runlengths:
#                 runlength_values = base_runlengths[key]
#                 all_runlengths_per_base[b][key].extend(runlength_values)
#
#         if n % 1 == 0:
#             sys.stdout.write("\r " + str(round(n/n_files*100,3)) + "% completed")
#
#             n += 1
#
#         # if n > 10000:
#         #     break
#
#     sys.stdout.write("\r 100%% completed     \n")
#
#     # concatenate observed runlength lists to a single array
#     for b, base_runlengths in enumerate(all_runlengths_per_base):
#         for key in base_runlengths:
#             runlength_values = base_runlengths[key]
#             all_runlengths_per_base[b][key] = numpy.concatenate(runlength_values)
#
#     plot_runlength_distributions(all_runlengths_per_base)
#
#     base_frequency_matrices = encode_runlength_distributions_as_matrix(all_runlengths_per_base,
#                                                                        log_scale=False,
#                                                                        normalize_observed=False)
#
#     for frequency_matrix in base_frequency_matrices:
#         pyplot.imshow(frequency_matrix)
#         pyplot.show()
#
#     output_dir = "output/runlength_frequency_matrix"
#     datetime_string = '-'.join(list(map(str, datetime.datetime.now().timetuple()))[:-3])
#     filename = "runlength_probability_matrix_" + datetime_string
#
#     save_numpy_matrices(output_dir=output_dir, filename=filename, matrices=base_frequency_matrices)
#
#
# def run():
#     # data_path = "/home/ryan/code/nanopore_assembly/output/celegans_chr1_1m_windows_spoa_pileup_generation_2018-9-18"    # 1 million bases in celegans chr1 scrappie
#     # data_path = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_human_chr1_1mbp_2018-9-18"             # 1 million bases in human guppy
#     # data_path = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_celegans_chr1_2-12Mbp_2018-9-21"       # 10 million bases in human guppy
#     data_path = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_chr5_FULL_20Mbp_2018-9-24"
#
#     file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=data_path, file_extension=".npz")
#
#     data_loader = DataLoader(file_paths, batch_size=1, parse_batches=False)
#
#     lengths = measure_runlengths(data_loader)
#
#
# if __name__ == "__main__":
#     run()