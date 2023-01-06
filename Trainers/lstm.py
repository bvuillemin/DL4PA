"""
Deep Learning Framework
Version 1.5
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

from __future__ import print_function, division
import tensorflow as tf
from math import ceil
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from Trainers import Trainer


class Mask(keras.layers.Layer):
    def call(self, inputs):
        return tf.experimental.numpy.any(tf.not_equal(inputs, 0), axis=2)


class LSTMTrainer(Trainer):
    def build(self, preparator, epoch_counter):
        super().build(preparator, epoch_counter)

    def core(self):

        # build the model:
        print('Build model...')

        main_input = layers.Input(
            shape=(self.preparator.orchestrator.max_case_length-1, self.preparator.orchestrator.features_counter),
            name='main_input', dtype='float32')
        mask = Mask()(main_input)
        # train a 2-layer LSTM with one shared layer
        l1 = layers.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(
            main_input, mask=mask)  # the shared layer
        b1 = layers.BatchNormalization()(l1)
        l2_1 = layers.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(
            b1)  # the layer specialized in activity prediction
        b2_1 = layers.BatchNormalization()(l2_1)
        act_output = layers.Dense(self.preparator.orchestrator.activity_counter, activation='softmax',
                           kernel_initializer='glorot_uniform', name='act_output')(
            b2_1)

        model = Model(inputs=[main_input], outputs=[act_output])

        opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

        model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=opt)

        early_stopping = EarlyStopping(monitor='val_loss', patience=50)
        model_checkpoint = ModelCheckpoint("Output/"+self.preparator.orchestrator.output_name+'/Trainers/model_{epoch:02d}-{loss:.2f}.h5', monitor='val_loss',
                                           verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)
        return model, [early_stopping, model_checkpoint, lr_reducer]

    def train_model_online(self):
        model, callbacks = self.core()
        epoch_size_train = self.preparator.get_epoch_size_online(0)
        epoch_size_val = self.preparator.get_epoch_size_online(1)
        model.fit(self.preparator.run_online(0), verbose=2, validation_data=self.preparator.run_online(1), callbacks=callbacks,
                  steps_per_epoch=ceil(epoch_size_train / self.preparator.batch_size), epochs=self.epoch_counter,
                  validation_steps=ceil(epoch_size_val / self.preparator.batch_size))
        self.model = model
        return model

    def train_model_offline(self):
        model, callbacks = self.core()
        epoch_size_train = self.preparator.get_epoch_size_offline(0)
        epoch_size_val = self.preparator.get_epoch_size_offline(1)
        model.fit(self.preparator.read_offline(0), verbose=1, validation_data=self.preparator.read_offline(1), callbacks=callbacks,
                  steps_per_epoch=ceil(epoch_size_train / self.preparator.batch_size), epochs=self.epoch_counter,
                  validation_steps=ceil(epoch_size_val / self.preparator.batch_size))
        self.model = model
        return model

    def load_model(self):
        model = keras.models.load_model("Output/" + self.preparator.orchestrator.output_name + '/Models/LSTM_model')
        self.model = model

    def save_model(self):
        self.model.save("Output/" + self.preparator.orchestrator.output_name + '/Models/LSTM_model')
