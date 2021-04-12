from cdcr.entities.entity_preprocessor import EntityPreprocessor
from cdcr.entities.identifier import Identifier
from cdcr.structures.document_set import DocumentSet
from cdcr.structures.entity_set import EntitySet
from cdcr.structures.entity import Entity
from cdcr.candidates.cand_enums import *


class CORENLPIdentifier(Identifier):

    """
    One of the baselines to show capability of CoreNLP to resolve mentions on combined documents
    """

    def __init__(self, document_set: DocumentSet):
        super().__init__(document_set)

        self.config = document_set.configuration.entity_identifier_config.params

    def extract_entities(self) -> EntitySet:
        if self.docs.candidates.coref_strategy != CorefStrategy.MULTI_DOC:
            raise ValueError("Candidate extraction of this document set is not MultiDocument. Rerun candidate extraction "
                             "module with \"coref_extraction_strategy = CorefStrategy.MULTI_DOC\" and then proceed with "
                             "CORENLP entity identifier method.")

        ent_preprocessor = EntityPreprocessor(self.docs, Entity)
        entity_dict = ent_preprocessor.entity_dict_construction()
        entity_set = EntitySet(identification_method=self.docs.configuration.entity_method, topic=self.docs.topic)
        entity_set.extend(list(entity_dict.values()))
        entity_set.sort(reverse=True, key=lambda x: len(x.members))
        return entity_set
