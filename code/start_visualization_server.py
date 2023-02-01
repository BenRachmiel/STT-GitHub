import graph_functions as gf
import multiprocessing as mp
import json


def initialize_new_data_json(filepath):
    data = {
        'Epoch': [],
        'WER': [],
        'CER': [],
        'Loss': [],
        'Loss2': []
    }
    with open(filepath, 'w') as fp:
        fp.seek(0)
        json.dump(data, fp)


def start_visualization(filepath, update_time_seconds=10, model_name='No Model Name Provided'):
    initialize_new_data_json(filepath)

    server_process = mp.Process(target=gf.dash_server, args=(filepath, update_time_seconds, model_name))
    server_process.start()
