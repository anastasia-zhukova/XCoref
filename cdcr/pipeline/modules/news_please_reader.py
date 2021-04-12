import json
import pickle

from cdcr.structures import DocumentSet
from cdcr.util.cache import Cache
from cdcr.structures import DocumentSet, Document, EmptyParseAttributeException
import cdcr.config as config
import os
import sys
import re
from typing import *

from cdcr.structures import DocumentSet


class NewsPleaseReader:
    """
    A module reads all documents in given a path provided in news please format. The
    path is read from the provided document set.
    """

    @staticmethod
    def run(document_set: DocumentSet, topic: str = None, config: Dict = None, pipeline_modules: List[str] = []) -> \
            DocumentSet:
        """
        Read documents in news please format from a given path.

        Args:
            document_set (DocumentSet or str): A DocSet or a path to the documents to read.

        Returns:
            document_set (DocumentSet): A DocSet including the processed documents.
        """
        document_set = UserInterface.run(topic, config, pipeline_modules)

        # read initial path from the passed DocSet
        path = document_set.processing_information.source_path

        articles = []
        for file in os.listdir(path):
            file = os.path.join(path, file)
            if file.endswith('json'):
                file = json.load(open(file, "r", encoding="utf8"))
            elif file.endswith('pickle'):
                file = pickle.load(open(file, "r"))
            elif os.path.isdir(file):
                continue
            articles.append(file)

        # add documents to the document set
        for article in articles:
            try:
                document = Document.from_news_please(article)
                document_set.append(document)
            except (EmptyParseAttributeException, AssertionError):
                raise ValueError("Document could not be loaded from the source_path.")

        return document_set


NOTIFICATION_MESSAGES = {
    # errors
    "no_folders": "\nNo folders with data found. Please place a data folder to {0} and run the code again.\n",
    "wrong_index": "\nThis index in existing {0} list doesn't exist. Try again.\n",
    "wrong_yes_no": "\nNo such restore reply possible. Enter yes(y) or no(n) answer.\n",
    "no_saved_topic": "\nNo saved data on such topic. The pipeline will start from the beginning.\n",

    # interactions
    "select_topic": "\nChoose topic id you want to work on:",
    "select_to_restore": "\nDo you want to restore data from a saved file (y, n):",
    "select_folder": "\nSelect index of a file to restore: ",
    "quit": "\nTo quit pipeline execution press Enter."
}


class UserInterface:
    """
    A class enabling user input of a topic to process and such operations as selection of which data
    to restore (or save).
    This class outputs two parameters: a path to a data folder to work with and a name of the task to start/restore the
    pipeline. These two variables are dependent because restoring the pipeline from some point required parameters from
    the previous steps to be present.
    The execution of this class could be omitted but the main run code will require the mentioned two parameters entered
    manually.
    """

    _yes_no_dic = {"y": True,
                   "n": False,
                   False: False}

    def __init__(self, topic: str = None):
        if not UserInterface.get_available_data_folders():
            sys.exit(NOTIFICATION_MESSAGES["no_folders"].format(config.ORIGINAL_DATA_PATH))

        self.topic = topic if topic is not None else self._select_topic()

        self.path = \
            UserInterface._get_full_folder_path(config.ORIGINAL_DATA_PATH, self.topic)

    def _select_topic(self):
        folders = sorted(UserInterface.get_available_data_folders())
        for i, d in enumerate(folders):
            print(str(i) + ": " + d)
        reply = UserInterface._check_range_reply(NOTIFICATION_MESSAGES["select_topic"],
                                                 [i for i in range(len(folders))],
                                                 NOTIFICATION_MESSAGES["wrong_index"].format("topics"))
        return folders[reply]

    @staticmethod
    def get_available_data_folders():
        for root, dirs, files in os.walk(config.ORIGINAL_DATA_PATH):
            return sorted(dirs)

    @staticmethod
    def _check_range_reply(input_message, possible_values, error_message):
        while True:
            reply = int(input(input_message))
            if reply not in possible_values:
                print(error_message)
            else:
                break
        return reply

    @staticmethod
    def _get_full_folder_path(path, data_folder):
        return os.path.join(path, data_folder)

    @classmethod
    def run(cls, topic: str = None, config: Dict = None, pipeline_modules: List[str] = []) -> DocumentSet:
        """
        Show a user interface to set initial processing information, e.g.: topic.

        Returns:
            document_set (DocumentSet): A DocSet including the initial processing information.

        """
        # initialize user interface
        input_data = cls(topic)

        # initialize empty document set
        document_set = DocumentSet(topic=input_data.topic, config=config, module_names=pipeline_modules)
        document_set.processing_information.source_path = input_data.path
        return document_set
