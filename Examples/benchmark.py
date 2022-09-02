"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

import time

from Managers.common_functions import *
from Managers.encoder_manager import *
from Editors import *
from Encoders import *
from DataPreparators import *
from Trainers import *
from memory_profiler import memory_usage


def benchmark(filename):
    input_path = "./Data/"+filename+".csv"
    output_name = filename
    editors = [SosForAll(), EosForAll()]

    encoders = [DeleteEncoder(0), OneHotEncoder(1, activity=True), TimeDifferenceSingleEncoder(2)]
    editor_manager = EditorManager(editors)
    encoder_manager = EncoderManager(encoders)

    start_time = time.time()
    orchestrator = build_orchestrator(input_path, output_name, input_chunk_size, encoder_manager, editor_manager, dates_ids, double_timestamps)
    preparator = SlicerLSTM()
    preparator.build(input_chunk_size, output_chunk_size, batch_size, orchestrator)
    lstm_trainer = LSTMTrainer()
    lstm_trainer.build(preparator, epoch_counter)
    init_time = time.time()
    total = ceil(lstm_trainer.preparator.get_epoch_size_online() / lstm_trainer.preparator.batch_size)

    # Loop for processing all the input data
    i = 0
    for _ in tqdm(preparator.run_online(), total=total):
        i += 1
        if i == total:
            break
    preprocessing_time = time.time()
    print("Initialization time =", init_time - start_time)
    print("Preprocessing time =", preprocessing_time - init_time)


# Please install memory_profiler before running this benchmark!
# First, if you haven't done so, please unzip Data.zip to the root of the project.


if __name__ == '__main__':
    # Change the filename with either "helpdesk", "env_permit", "bpi_12_w", "bpi_13_incidents", "bpi_17", or "bpi_19"
    filename = "helpdesk"
    input_chunk_size = 5000
    output_chunk_size = 5000
    batch_size = 64
    epoch_counter = 5
    double_timestamps = False
    dates_ids = [2]
    print(filename)
    mem_usage = memory_usage((benchmark, [filename],))
    print('Maximum memory usage:', max(mem_usage))
