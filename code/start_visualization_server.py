import graph_functions as gf
import multiprocessing as mp
import json


def initialize_new_data_json(filepath, datasetid):
    data = {
        'Epoch': [],
        'Epoch_valid': [],
        'WER': [],
        'WER_valid': [],
        'CER': [],
        'CER_valid': [],
        'Loss': [],
        'Loss_valid': []
    }
    true_filepath = filepath.replace('.json', f'_id_{datasetid}.json')
    with open(true_filepath, 'w') as fp:
        fp.seek(0)
        json.dump(data, fp)


def start_visualization(passed_args, update_time_seconds=10, model_name='No Model Name Provided'):
    server_process = mp.Process(target=gf.dash_server, args=(passed_args, update_time_seconds, model_name))
    server_process.start()
