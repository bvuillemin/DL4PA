"""
Deep Learning Framework
Version 1.5
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""
from math import ceil
import numpy as np


class Trainer:
    def __init__(self) -> None:
        """
        Creates a Trainer object

        """
        super().__init__()
        self.preparator = None
        self.epoch_counter = None
        self.model = None

    def build(self, preparator, epoch_counter):
        """
        Sets the internal properties of the Trainer

        :param preparator: Data preparator
        :type preparator: DataPreparator
        :param epoch_counter: Number of inputs per epoch
        :type epoch_counter: int
        """
        self.preparator = preparator
        self.epoch_counter = epoch_counter

    def train_model_online(self):
        """
        Trains the neural network online

        """
        pass

    def train_model_offline(self):
        """
        Trains the neural network offline

        """
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass
