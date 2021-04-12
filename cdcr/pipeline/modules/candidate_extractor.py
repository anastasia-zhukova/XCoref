import time
# from newsalyze.candidates._archive.fivew_phrases_extractor import FiveWPhrasesExtractor
# from newsalyze.candidates._archive.np_extractor import NPExtractor
# from newsalyze.candidates._archive.freq_phrases_extractor import FreqPhrasesExtractor
# from newsalyze.candidates._archive.wordvectors import Wordvectors
# from newsalyze.candidates._archive.word2vec_dim import Word2VecDimensions
# from newsalyze.candidates._archive.glove_dim import GloveDimensions
# from newsalyze.candidates.coded_segments import CodedSegmentsExtractor
from cdcr.candidates.extract_candidates import CandidatePhrasesExtractor

from cdcr.structures import DocumentSet


NOTIFICATION_MESSAGES = {
    "cand_extr": "Extracting candidates from \"{0}\".",
    "no_data": "No input data read for the Candidate Extractor module."
}

# CURRENT_EXTRACTOR = "coref_np"
# CURRENT_EXTRACTOR = "coded"
# CURRENT_EXTRACTOR = "global"
#
# CURRENT_WORDVECTORS = "word2vec"



class CandidateExtractor:
    """
    A CandidateExtractor class performs extraction of small lexical units (group of words) that will play a role of
    framing devices candidates. It executes tasks for extraction of candidates of different types. In order to add or
    remove tasks one needs to adjust _extractors.
    """

    # _extractors = {
    #     # "np": NPExtractor(),
    #     "coref_np": CorefsAndNPPhrasesExtractor(),
    #     # TODO change code in the following extractors according to the new changes for further evaluation
    #     # "fivew": FiveWPhrasesExtractor(),
    #     # "freq": FreqPhrasesExtractor(),
    #     # "coded": CodedSegmentsExtractor(),
    #     "global": GlobalCorefPhrasesExtractor()
    # }

    def __init__(self, module_name):
        self.module_name = module_name

    @classmethod
    def run(cls, document_set: DocumentSet) -> DocumentSet:
        """
        Extract candidates from documents in the given DocSet.

        Args:
            document_set (DocumentSet): A DocSet including the documents to apply candidate extraction.

        Returns:
            document_set (DocumentSet): A DocSet including the extracted candidates for each document.
        """

        # print(CURRENT_EXTRACTOR)
        # candidates = cls._extractors[CURRENT_EXTRACTOR].extract_phrases(document_set)
        start_time = time.time()

        candidates = CandidatePhrasesExtractor().extract_phrases(document_set)

        end_time = time.time()
        document_set.processing_information.cand_execution_time = end_time - start_time
        # document_set.candidates = [{key: cands} for key, cands in list(candidates.items())]
        document_set.candidates = candidates

        return document_set
