import cdcr.config as config
from cdcr.util.reader import Reader

import os, re

document_storage = {}

class LoadDocs:

    def __init__(self):
        self._reader = Reader()

    def load_docs(self, topic, reload=False):
        if topic in document_storage and not reload:
            return document_storage[topic]
        listdir = sorted(os.listdir(config.SAVED_DATA_PATH))
        # TODO load docs not from a last preprocc folder but the one that has an id of the current pipeline process
        doc_folder_id = sorted([i for i, item in enumerate(listdir)
                         if re.search('preprocessing', item) and re.search(topic, item)])[-1]
        docs = self._reader.read_data(os.path.join(config.SAVED_DATA_PATH, listdir[doc_folder_id]))
        document_storage[topic] = docs
        return docs
