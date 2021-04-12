import os, json, pickle, logging
import cdcr.config as config
from cdcr.logger import LOGGER


class Reader:
    """
    A helper class to read data.
    """

    _logger = LOGGER

    def __init__(self):
        self._data = None

    def read_data(self, data):
        if isinstance(data, list):
            self._data = data
        elif isinstance(data, str):
            self._data = _read_files(data)
        return self._data

    def get_data(self):
        return self._data


def _read_files(data_path):
    output = []
    for f in os.listdir(data_path):
        if f.endswith('.json'):
            data = _read_json_file(os.path.join(data_path, f))
        elif f.endswith('.pickle'):
            data = _read_pickle_file(os.path.join(data_path, f))
        else:
            data = None
        output.append(data)
    if len(output) is 1:
        return output[0]
    return output


def _read_json_file(path):
    with open(path, 'r', encoding='utf8') as input_file:
        data = json.load(input_file)
        input_file.close()
    return data


def _read_pickle_file(path):
    return pickle.load(open(path, 'rb'))
