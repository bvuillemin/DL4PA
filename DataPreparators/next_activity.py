"""
Deep Learning Framework
Version 1.5
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from DataPreparators.base_data_prep import *
from Managers.encoder_manager import EncoderManager
from generic_functions import get_total_chunk_number


class NextActivity(DataPreparator):
    def __init__(self):
        super().__init__()
        self.list = []
        self.dict_size = {}

    def build(self, input_chunk_size, output_chunk_size, batch_size, orchestrator):
        """
        Assigns the necessary information to the preparator

        :param input_chunk_size: Size of a chunk (number of cases inside a chunk)
        :type input_chunk_size: int
        :param output_chunk_size: Number of cases per chunk
        :type output_chunk_size: int
        :param batch_size: Size of a batch to train the neural network
        :type batch_size: int
        :param orchestrator: Orchestrator
        :type orchestrator: Orchestrator
        """
        super().build(input_chunk_size, output_chunk_size, batch_size, orchestrator)
        # Create the Train (0), Validation (1) and Test (2) sets with their respective probabilities
        values = (0, 1, 2)
        probabilities = (0.7, 0.2, 0.1)
        size = self.get_epoch_size_online()
        # Randomly separate data into Train, Validation and Test sets
        self.list = np.array(random.choices(values, probabilities, k=size))
        self.dict_size[0] = self.get_epoch_size_offline(0)
        self.dict_size[1] = self.get_epoch_size_offline(1)
        self.dict_size[2] = self.get_epoch_size_offline(2)
        # Create the output decoders
        # Get the first OneHot decoder, that is used for activities
        activity_decoder = None
        for encoder in self.input_decoders.encoders:
            if encoder.name == "OneHot":
                activity_decoder = encoder
                break
        if activity_decoder is None:
            raise RuntimeError('No activity encoder is present, thus preventing the Data Preparator to operate')
        self.output_decoders = EncoderManager([activity_decoder])

    def run_online(self, value=None, get_leftovers=False):
        """
        Slices cases into suffixes and prefixes (online mode)

        """
        while True:
            iterator = 0
            prefixes = []
            suffixes = []
            if get_leftovers:
                all_leftovers = []
                leftovers_names = self.orchestrator.encoder_manager.get_leftover_names()
            for case, leftover in self.orchestrator.process_online(self.input_chunk_size):
                if value is None or self.list[iterator] == value:
                    i = 1
                    while i < len(case):
                        prefixes.append(case[:i])
                        suffixes.append(case[i, :self.orchestrator.activity_counter])
                        if get_leftovers:
                            all_leftovers.append(leftover)
                        i += 1
                        if len(prefixes) == self.batch_size:
                            prefixes = tf.keras.preprocessing.sequence.pad_sequences(prefixes, padding='post',
                                                                                     maxlen=self.orchestrator.max_case_length-1,
                                                                                     dtype=float)
                            suffixes = np.array(suffixes, dtype=float)
                            if get_leftovers:
                                all_leftovers = pd.DataFrame(data=all_leftovers, columns=leftovers_names)
                                yield prefixes, suffixes, all_leftovers
                            else:
                                yield prefixes, suffixes
                            prefixes = []
                            suffixes = []
                            if get_leftovers:
                                all_leftovers = []
                iterator += 1
            if len(prefixes) > 0:
                prefixes = tf.keras.preprocessing.sequence.pad_sequences(prefixes, padding='post',
                                                                         maxlen=self.orchestrator.max_case_length-1,
                                                                         dtype=float)
                suffixes = np.array(suffixes, dtype=float)
                if get_leftovers:
                    all_leftovers = pd.DataFrame(data=all_leftovers, columns=leftovers_names)
                    yield prefixes, suffixes, all_leftovers
                else:
                    yield prefixes, suffixes

    def get_epoch_size_online(self, value=None):
        if value is None:
            total = 0
            for cases in self.orchestrator.edit_online(self.input_chunk_size, self.output_chunk_size):
                for case in cases:
                    total += len(case) - 1
            print(total, "cases")
        else:
            total = np.count_nonzero(self.list == value)
        return total

    def run_offline(self):
        write_attribute = "wb"
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
                        prefixes.append(case[:i])
                        suffixes.append(case[i, :self.orchestrator.activity_counter])
                        i += 1
                        if len(prefixes) == self.output_chunk_size or case_counter == self.orchestrator.case_counter:
                            # Write the results!
                            # If we are in the first chunk, create a new file. Else, append
                            with open("Output/" + self.orchestrator.output_name +  "/prefixes.npy",
                                      write_attribute) as output_file:
                                for prefix in prefixes:
                                    np.save(output_file, prefix)
                            with open("Output/" + self.orchestrator.output_name +  "/suffixes.npy",
                                      write_attribute) as output_file:
                                for suffix in suffixes:
                                    np.save(output_file, suffix)
                            if write_attribute == "wb":
                                write_attribute = "ab"
                            prefixes = []
                            suffixes = []

    def read_offline(self, value=None):
        """
        Slices cases into suffixes and prefixes (online mode)

        """
        with open("Output/" + self.orchestrator.output_name + "/prefixes.npy", 'rb') as input_file1:
            with open("Output/" + self.orchestrator.output_name + "/suffixes.npy", 'rb') as input_file2:
                epoch_size = self.get_epoch_size_offline(value)
                while True:
                    iterator = 0
                    for chunk_index in range(
                            get_total_chunk_number(epoch_size, self.batch_size)):
                        prefixes = []
                        suffixes = []
                        counter = 0
                        if chunk_index == get_total_chunk_number(epoch_size, self.batch_size) - 1:
                            chunk_range = epoch_size % self.batch_size
                        # Else, the number of cases if equal of the size of a chunk
                        else:
                            chunk_range = self.batch_size
                        while counter < chunk_range:
                            if value is None or self.list[iterator] == value:
                                prefixes.append(np.load(input_file1, allow_pickle=True))
                                suffixes.append(np.load(input_file2, allow_pickle=True))
                                counter += 1
                            iterator += 1
                        prefixes = tf.keras.preprocessing.sequence.pad_sequences(prefixes, padding='post',
                                                                                 maxlen=self.orchestrator.max_case_length-1,
                                                                                 dtype=float)
                        suffixes = np.array(suffixes)
                        yield prefixes, suffixes
                    input_file1.seek(0)
                    input_file2.seek(0)

    def get_epoch_size_offline(self, value=None):
        if value is None:
            total = 0
            with open("Output/" + self.orchestrator.output_name + "/suffixes.npy", 'rb') as input_file:
                while True:
                    try:
                        np.load(input_file, allow_pickle=True)
                        total += 1
                    except:
                        break
            print(total, "cases")
        else:
            total = np.count_nonzero(self.list == value)
        return total

    def decode_single_input(self, input):
        return self.input_decoders.encode_case(input[~np.all(input == 0, axis=1)])

    def decode_single_output(self, output):
        raw_result = self.output_decoders.encode_case(output)
        sos_indices = np.where(raw_result == "SoS")
        if len(sos_indices) == 0 or len(sos_indices[0]) == 0:
            return raw_result
        else:
            cut = sos_indices[0][-1]
            return raw_result[cut:]

