from contextlib import suppress

from graphene import ObjectType, String, List

from cdcr.structures import Document
from cdcr.structures.one_to_n_set import OneToNSet
from typing import Any, Iterable, Dict
from cdcr.structures.configuration import Configuration


class ProcessingInformation:
    def __init__(self, module_name, **kwargs):
        self.module_names = module_name
        # self.step_for_restoring = step_for_restoring
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def last_module_timestamp(self):
        return self.last_module_time.isoformat()


class DocumentSet(OneToNSet):
    """
    A set of documents related to one another.
    """

    CHILDRENS_PARENT_ATTRIBUTE = "document_set"

    def document_by_id(self, id_):
        for document in self:
            with suppress(AttributeError):
                if document.id == id_:
                    return document

    # def __init__(self, topic: str, step_for_restoring: str, module_names: Iterable = [], items: Iterable = [], force: bool = False):
    def __init__(self, topic, module_names: Iterable = [], items: Iterable = [], force: bool = False, config: Dict = None):
        """
        A set of documents that is passed through the pipeline. The document set holds all relevant data. The data gets
        saved, changed and deleted directly in the document set.
        """
        super().__init__(items, force)

        self.entities = None
        self.candidates = None
        self.visualizations = None
        self.processing_information = ProcessingInformation(module_names)
        self.topic = topic
        self.configuration = Configuration(self, config=config)

    def _test_and_parse(self, document: Any, force: bool = False) -> Document:
        """
        Adding that a documents gets parsed to a Document() if it is none already.

        See ``structures.one_to_n_set.OneToNSet._test_and_parse()``.
        """
        if type(document) is not Document:
            document = Document(document, self)
        return super()._test_and_parse(document, force)
