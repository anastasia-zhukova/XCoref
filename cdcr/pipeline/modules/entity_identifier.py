import datetime

from cdcr.entities.eecdcr.EECDCRIdentifier import EECDCRIdentifier
from cdcr.structures import DocumentSet
from cdcr.entities.sieve_based.tca_improved.tca_improved import TargetConceptAnalysisImproved
from cdcr.entities.sieve_based.xcoref.xcoref import XCoref
from cdcr.entities.sieve_based.xcoref_base.xcoref_base import XCoref_Base
from cdcr.entities.multidoc_corenlp.corenlp_identifier import CORENLPIdentifier
from cdcr.entities.clustering.clustering_identifier import ClusteringIdentifier
from cdcr.entities.sieve_based.tca_orig.tca_orig import TargetConceptAnalysisOriginal
from cdcr.entities.lemma.lemma_merge import LemmaIdentifier
from cdcr.entities.const_dict_global import *


class EntityIdentifier:
    """
    A module identifies the same concepts (actors, actions, places, etc.) but which throughout the analysed texts have
    mentioned throughout the analyzed text and have same or different wording.
    """

    def __init__(self, module_name):
        self.module_name = module_name

        self.identifiers = {
            TCA_ORIG: TargetConceptAnalysisOriginal,
            TCA_IMPROVED: TargetConceptAnalysisImproved,
            XCOREF: XCoref,
            XCOREF_BASE: XCoref_Base,
            EECDCR: EECDCRIdentifier,
            CORENLP: CORENLPIdentifier,
            CLUSTERING: ClusteringIdentifier,
            LEMMA: LemmaIdentifier
        }

    @staticmethod
    def run(document_set: DocumentSet) -> DocumentSet:
        """
        Identify entities based on the candidates in each document of the given DocSet.

        Args:
            document_set (DocumentSet): A DocSet including the documents with extracted candidates.

        Returns:
            document_set (DocumentSet): A DocSet including the identified entities.
        """
        start_time = datetime.datetime.now()

        entity_identifier_module = EntityIdentifier(document_set.topic).identifiers[document_set.configuration.entity_method]
        identifier = entity_identifier_module(document_set)
        document_set.entities = identifier.extract_entities()

        end_time = datetime.datetime.now()
        document_set.processing_information.entity_execution_time = end_time - start_time
        return document_set
