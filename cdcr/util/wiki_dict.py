import pickle
import os
from datetime import datetime

from cdcr.config import RESOURCES_PATH


def init():
    global _wiki_dict
    _wiki_dict = None
    for path in os.listdir(os.path.join(RESOURCES_PATH, "wiki_dict"))[::-1]:
        with open(os.path.join(RESOURCES_PATH, "wiki_dict", path), "rb") as file:
            w_dict = pickle.load(file)
            if not len(w_dict) and len(os.listdir(os.path.join(RESOURCES_PATH, "wiki_dict"))) > 1:
                # in case the last file was saved empty the last time
                continue
            else:
                _wiki_dict = w_dict
                break

    if _wiki_dict is None:
        with open(os.path.join(RESOURCES_PATH, "wiki_dict", os.listdir(os.path.join(RESOURCES_PATH, "wiki_dict"))[-1]),
                  "rb") as file:
            _wiki_dict = pickle.load(file)


def load():
    return _wiki_dict


def update(new_wiki_dict):
    now = datetime.now()
    with open(os.path.join(RESOURCES_PATH, "wiki_dict", now.strftime("%Y-%m-%d") + "_" + "wiki_dict.pickle"), "wb") as file:
        pickle.dump(new_wiki_dict, file)
    _wiki_dict = new_wiki_dict
