"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from DataPreparators.base_data_prep import *
from generic_functions import get_total_chunk_number


class SlicerLSTM(DataPreparator):

    def run_online(self, get_leftovers=False):
        """
        Slices cases into suffixes and prefixes (online mode)

        """
        while True:
            prefixes = []
            suffixes = []
            if get_leftovers:
                all_leftovers = []
                leftovers_names = self.orchestrator.encoder_manager.get_leftover_names()
            for case, leftover in self.orchestrator.process_online(self.input_chunk_size):
                i = 1
                while i < len(case):
                    prefixes.append(self.pad(case[:i]))
                    suffixes.append(case[i, :self.orchestrator.activity_counter])
                    if get_leftovers:
                        all_leftovers.append(leftover)
                    i += 1
                    if len(prefixes) == self.batch_size:
                        prefixes = np.asarray(prefixes).astype(float)
                        suffixes = np.asarray(suffixes).astype(float)
                        if get_leftovers:
                            all_leftovers = pd.DataFrame(data=all_leftovers, columns=leftovers_names)
                            yield prefixes, suffixes, all_leftovers
                        else:
                            yield prefixes, suffixes
                        prefixes = []
                        suffixes = []
                        if get_leftovers:
                            all_leftovers = []
            if len(prefixes) > 0:
                prefixes = np.asarray(prefixes).astype(float)
                suffixes = np.asarray(suffixes).astype(float)
                if get_leftovers:
                    all_leftovers = pd.DataFrame(data=all_leftovers, columns=leftovers_names)
                    yield prefixes, suffixes, all_leftovers
                else:
                    yield prefixes, suffixes

    def get_epoch_size_online(self):
        total = 0
        for cases in self.orchestrator.edit_online(self.input_chunk_size, self.output_chunk_size):
            for case in cases:
                total += len(case) - 1
        print(total, "cases")
        return total

    def run_offline(self):
        first_chunk = True
        with open("Output/" + self.orchestrator.output_name + "/data.npy", 'rb') as input_file:
            with tqdm(total=self.orchestrator.case_counter, desc='Slice into prefix/suffix') as pbar:
                case_counter = 0
                prefixes = []
                suffixes = []
                # Add all the cases to a list
                while case_counter < self.orchestrator.case_counter:
                    case = np.load(input_file, allow_pickle=True)
                    case_counter += 1
                    pbar.update(1)
                    i = 1
                    while i < len(case):
                        prefixes.append(self.pad(case[:i]))
                        suffixes.append(case[i, :self.orchestrator.activity_counter])
                        i += 1
                        if len(prefixes) == self.output_chunk_size or case_counter == self.orchestrator.case_counter:
                            prefixes = np.asarray(prefixes, dtype=float)
                            suffixes = np.asarray(suffixes, dtype=float)
                            # Write the results!
                            # If we are in the first chunk, create a new file
                            if first_chunk:
                                with open("Output/" + self.orchestrator.output_name + "/prefixes.npy",
                                          'wb') as output_file:
                                    for prefix in prefixes:
                                        np.save(output_file, prefix)
                                with open("Output/" + self.orchestrator.output_name + "/suffixes.npy",
                                          'wb') as output_file:
                                    for suffix in suffixes:
                                        np.save(output_file, suffix)
                                first_chunk = False
                            # Else, append to this file
                            else:
                                with open("Output/" + self.orchestrator.output_name + "/prefixes.npy",
                                          'ab') as output_file:
                                    for prefix in prefixes:
                                        np.save(output_file, prefix)
                                with open("Output/" + self.orchestrator.output_name + "/suffixes.npy",
                                          'ab') as output_file:
                                    for suffix in suffixes:
                                        np.save(output_file, suffix)
                            prefixes = []
                            suffixes = []

    def read_offline(self):
        """
        Slices cases into suffixes and prefixes (online mode)

        """
        with open("Output/" + self.orchestrator.output_name + "/prefixes.npy", 'rb') as input_file1:
            with open("Output/" + self.orchestrator.output_name + "/suffixes.npy", 'rb') as input_file2:
                while True:
                    for chunk_index in range(
                            get_total_chunk_number(self.orchestrator.case_counter, self.batch_size)):
                        prefixes = []
                        suffixes = []
                        if chunk_index == get_total_chunk_number(self.orchestrator.case_counter,
                                                                 self.batch_size) - 1:
                            chunk_range = self.orchestrator.case_counter % self.batch_size
                        # Else, the number of cases if equal of the size of a chunk
                        else:
                            chunk_range = self.batch_size
                        for _ in range(chunk_range):
                            prefixes.append(np.load(input_file1, allow_pickle=True))
                            suffixes.append(np.load(input_file2, allow_pickle=True))
                        prefixes = np.asarray(prefixes)
                        suffixes = np.asarray(suffixes)
                        yield prefixes, suffixes
                    input_file1.seek(0)
                    input_file2.seek(0)

    def get_epoch_size_offline(self):
        total = 0
        with open("Output/" + self.orchestrator.output_name + "/suffixes.npy", 'rb') as input_file:
            while True:
                try:
                    np.load(input_file, allow_pickle=True)
                    total += 1
                except:
                    break
        print(total, "cases")
        return total

    def pad(self, case):
        return np.pad(case, [(self.orchestrator.max_case_length - len(case) - 1, 0), (0, 0)])

