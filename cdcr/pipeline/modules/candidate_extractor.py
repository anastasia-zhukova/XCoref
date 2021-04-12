import time
from cdcr.candidates.extract_candidates import CandidatePhrasesExtractor

from cdcr.structures import DocumentSet


NOTIFICATION_MESSAGES = {
    "cand_extr": "Extracting candidates from \"{0}\".",
    "no_data": "No input data read for the Candidate Extractor module."
}


class CandidateExtractor:
    """
    A CandidateExtractor class performs extraction of small lexical units (group of words) that will play a role of
    framing devices candidates. It executes tasks for extraction of candidates of different types. In order to add or
    remove tasks one needs to adjust _extractors.
    """

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
