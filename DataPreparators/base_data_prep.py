"""
Deep Learning Framework
Version 1.5
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""
from Managers.common_functions import create_all_decoders


class DataPreparator:

    def __init__(self) -> None:
        """
        Initializes the DataPreparator object

        """
        self.input_chunk_size = None
        self.output_chunk_size = None
        self.batch_size = None
        self.orchestrator = None
        self.input_decoders = None
        self.output_decoders = None

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
        self.input_chunk_size = input_chunk_size
        self.output_chunk_size = output_chunk_size
        self.batch_size = batch_size
        self.orchestrator = orchestrator
        self.input_decoders = create_all_decoders(orchestrator)

    def run_online(self, get_leftovers=False):
        """
        Separates the encoded data into inputs and outputs (online mode)

        """
        pass

    def get_epoch_size_online(self):
        """
        Computes the size of an epoch (online mode)

        """
        pass

    def run_offline(self):
        """
        Separates the encoded data into inputs and outputs (offline mode)

        """
        pass

    def get_epoch_size_offline(self):
        """
        Computes the size of an epoch (offline mode)

        """
        pass

    def decode_single_input(self, input):
        return self.input_decoders.encode_case(input)

    def decode_inputs(self, inputs, leftovers):
        if not leftovers.empty:
            for encoder in self.input_decoders.encoders:
                if encoder.leftover_name:
                    info = leftovers[encoder.leftover_name]
                    encoder.set_leftover(info)
        return [self.decode_single_input(input) for input in inputs]

    def decode_single_output(self, output):
        return self.output_decoders.encode_case(output)

    def decode_outputs(self, outputs, leftovers):
        if not leftovers.empty:
            for encoder in self.output_decoders.encoders:
                if encoder.leftover_name:
                    info = leftovers[encoder.leftover_name]
                    encoder.set_leftover(info)
        return [self.decode_single_output(output) for output in outputs]
