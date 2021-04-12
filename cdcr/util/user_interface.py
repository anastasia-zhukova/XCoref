import cdcr.config as config
import os
import re
import sys

from cdcr.structures import DocumentSet


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

    def __init__(self):
        if not UserInterface._get_available_folders():
            sys.exit(NOTIFICATION_MESSAGES["no_folders"].format(config.ORIGINAL_DATA_PATH))

        self._select_topic()

        self.path = \
            UserInterface._get_full_folder_path(config.ORIGINAL_DATA_PATH, self.topic)

    def _select_topic(self):
        folders = sorted(UserInterface._get_available_folders())
        for i, d in enumerate(folders):
            print(str(i) + ": " + d)
        reply = UserInterface._check_range_reply(NOTIFICATION_MESSAGES["select_topic"],
                                                 [i for i in range(len(folders))],
                                                 NOTIFICATION_MESSAGES["wrong_index"].format("topics"))
        self.topic = folders[reply]

    @staticmethod
    def _get_available_folders():
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
    def run(cls, module_names) -> DocumentSet:
        """
        Show a user interface to set initial processing information, e.g.: topic.

        Returns:
            document_set (DocumentSet): A DocSet including the initial processing information.

        """
        # initialize user interface
        input_data = cls()

        # initialize empty document set
        # document_set = DocumentSet(topic=input_data.get_topic(),
        #                            step_for_restoring=input_data.get_step() if input_data.get_step() else False,
        document_set = DocumentSet(input_data=input_data,
                                   module_names=module_names)
        # document_set.processing_information.source_path = input_data.get_path()
        # if input_data.get_step():
        #     document_set.processing_information.step_for_restoring = input_data.get_step()
        # else:
        #     document_set.processing_information.step_for_restoring = False
        return document_set
