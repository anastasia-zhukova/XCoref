import cdcr.config as config
from pycorenlp import StanfordCoreNLP
import string
import logging

from cdcr.logger import LOGGER
from cdcr.structures import Document
from stanfordnlp.server import CoreNLPClient
from CoreNLP_pb2 import Document as ProtobufDocument


NOTIFICATION_MESSAGES = {
    "start_corenlp": "The StanfordCoreNLP is not connected. Start the server (java -mx6g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer "
                "-port 9200 -timeout 90000) and press enter: ",
    "no_text": "No text provided for the annotation. ",
    "no_corenlp": "No corenlp output for \"{0}\". ",
    "run_time": "Restart your CoreNLP server with the larger timeout, e.g., 200000 (java -mx6g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer "
                "-port 9200 -timeout 200000). Press enter when finished to repeat preprocessing of this test:",
    "no_coref": "No method for coreference resolution is provided. Neural networks will be used as a default option. "
    }

# TODO java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9200 -timeout 300000

class CoreNLPCaller:

    NLP_CLIENT_SETTINGS = {
        "start_server": False,
        "properties": {
            "coref.algorithm": "neural"
        }
    }

    _logger = LOGGER

    def __init__(self):
        self._nlp = StanfordCoreNLP(config.CORENLP_SERVER)

    def combine_text_parts(self, doc):
        return doc.fulltext + " "

    def execute(self, text, annotators, coref="", with_wrapper=True, **corenlp_settings):
        try_again = True
        output = None

        while try_again:
            try:
                if len(text) > 0:
                    corenlp_settings = { **self.NLP_CLIENT_SETTINGS, **corenlp_settings }
                    corenlp_settings["annotators"] = annotators
                    with CoreNLPClient(**corenlp_settings, timeout=360*1000) as client:
                        output = client.annotate(text)
                    try_again = True
                else:
                    self._logger.warning(NOTIFICATION_MESSAGES["no_text"])

                if type(output) is not ProtobufDocument:
                    self._logger.error(NOTIFICATION_MESSAGES["no_corenlp"].format(text[:30] + "..."))
                    self._logger.error("\"" + output + "\"")
                    input(NOTIFICATION_MESSAGES["runtime"])
                else:
                    try_again = False

            except Exception:
                input(NOTIFICATION_MESSAGES["start_corenlp"])

        if with_wrapper:
            output = Document(output)
        return output
