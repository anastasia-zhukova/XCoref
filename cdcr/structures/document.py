"""Document containing the data of a via Stanford CoreNLP tagged document."""
import functools
from enum import Enum
import random

from graphene import ObjectType, Field, String, Interface
from graphene.utils.orderedtype import OrderedType

from cdcr.structures.coref import Coref
from cdcr.structures.coref_set import CorefSet
from cdcr.structures.political_bias import PoliticalSide, PoliticalBias
from cdcr.structures.sentence_set import SentenceSet
from cdcr.structures.dependencies import DependencyGraph
from cdcr.structures.ner_mention import NERMention
from typing import Optional, List, Iterator, Any, Dict
from graphene import List as GQLList
from stanfordnlp.server import CoreNLPClient
from stanfordnlp.protobuf.CoreNLP_pb2 import Token as ProtobufToken, Document as ProtobufDocument, DESCRIPTOR
import string
from cdcr.structures.wrapper import Wrapper
from cdcr.structures.tree import ParseTree
from contextlib import suppress
from cdcr.util.graphene import ObjectTypeProperty, FieldProperty
from cdcr.structures.graphene_schema import SCHEMA


class EmptyParseAttributeException(Exception):
    """
    Raised when an attribute is empty which is required not to be empty in order
    to be parsed for a method call.
    """
    pass


class Document(Wrapper):

    DEFAULT_NLP_CLIENT_SETTINGS = {
        "start_server": False,
        "annotators": "tokenize ssplit pos depparse parse ner coref".split(),
        "properties": {
            "coref.algorithm": "neural"
        }
    }

    JOIN_TEXTS_STRING = "\n"
    """str: How title, description and text should be joined when displayed when merged together."""

    TEXT_DEFAULT_ENDING = "."
    """str: How title, description and text should end when they do not end with a punctuation and are joined."""

    FULLTEXT_ATTRIBUTES = ["title", "description", "text"]
    """list(str): Attributes which contain text belonging to the full-text."""

    ALLOW_OVERWRITES = ["title", "description", "text", "maintext"]
    """list(str): Attributes to allow overwrites by external parsers (mostly setters which are detected by hasattr())."""

    DEFAULT_MAPPING = {"title": "title", "text": "text", "description": "description"}
    """dict(str, str): Default mapping for Document.from_object()."""

    WRAPPED_ATTRIBUTE = "_document"

    def __init__(self, document: Optional[ProtobufDocument] = None, original: Any = None):
        """
        Document containing the data of a via Stanford CoreNLP tagged document.
        Args:
            document: The (protobuf) response of ``stanfordnlp.server.CoreNLPClient.annotate``.
        """

        self.corefs = None
        """CorefSet: Coreferences this document contains."""

        self.__cached_fulltext = None
        """str: Title + Description + Text joined together."""

        self._title = ""
        """str: The title of the article."""

        self._description = None
        """Optional[str]: The description/intro-text of the article."""

        self._text = ""
        """str: The text of the article."""

        self.document_set = None
        """DocumentSet: (Parent of this object.) Set of documents this document is contained in."""

        self.document = document
        """ProtobufDocument: Stanford CoreNLP protobuf answer this document represents."""

        self.original = original
        """Any: Original item this document was generated from."""

        self.representativeness = 0
        """float: Representativeness of the document in the document-set. Default: 0."""

        self.__fulltext_offsets = {}
        """Dict[str, int]: The offsets from the start of the fulltext of each fulltext-attribute."""

        self.__id = None
        """Any: An ID which was set manually. If not set, self.id will return the index."""

    def __repr__(self):
        return self.source_domain + "_" + str(self.id)

    @property
    def fulltext_offsets(self):
        if not len(self.__fulltext_offsets):
            _ = self.fulltext  # This property calculates the offsets.
        return self.__fulltext_offsets

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title.strip()

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, description):
        if description:
            description = description.strip()
            self._description = description

    @property
    def text(self):
        return self._text

    @property
    def maintext(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text.strip()

    @maintext.setter
    def maintext(self, text):
        self._text = text.strip()

    @property
    def document(self) -> ProtobufDocument:
        """ProtobufDocument: Stanford CoreNLP protobuf answer this document represents."""
        return self._document

    @document.setter
    def document(self, document: Optional[ProtobufDocument]):
        self._document = document
        if document:
            self.corefs = CorefSet(self)
            self.sentences = SentenceSet(self)
            self.text = self.text or document.text

    @property
    def id(self):
        return self.__id or self.index

    @id.setter
    def id(self, id_):
        self.__id = id_

    @property
    def index(self) -> Optional[int]:
        """Index of this document in it's corresponding document set. ``None`` if not contained in a set."""
        if not self.document_set:
            return None
        return self.document_set.index(self)

    @property
    def dummy_political_bias(self) -> PoliticalBias:
        random.seed(len(self.text))
        side = random.choice(list(PoliticalSide))
        probability = random.randint(50, 100) / 100
        return PoliticalBias(side, probability)

    @property
    def dummy_representativeness(self) -> float:
        random.seed(len(self.text))
        return random.randint(33, 100) / 100

    @property
    def fulltext(self) -> str:
        """
        Title, Description and Text joined together. Checked for punctuation at the end of each sentence
        (``self.TEXT_DEFAULT_ENDING`` added if no punctuation was found).
        Joined by ``self.JOIN_TEXTS_STRING``.
        """
        if self.__cached_fulltext:
            return self.__cached_fulltext

        texts = []
        attributes = []
        for attribute in self.FULLTEXT_ATTRIBUTES:
            with suppress(AttributeError):
                text = getattr(self, attribute)
                if type(text) == str and text:
                    texts.append(text)
                    attributes.append(attribute)

        text_offsets = (len(texts)+1) * [0]
        for i in range(len(texts)):
            # Add TEXT_DEFAULT_ENDING if necessary.
            # This is done here (and not in the setter) as headlines do not include a fullstop, but
            # it is required for correct cross sentence entity detection.
            if texts[i][-1] not in string.punctuation:
                texts[i] += Document.TEXT_DEFAULT_ENDING
            text_offsets[i+1] = len(texts[i]) + text_offsets[i] + len(self.JOIN_TEXTS_STRING)

        fulltext = self.JOIN_TEXTS_STRING.join(texts)
        self.__fulltext_offsets = {key: text_offsets[i] for i, key in enumerate(attributes)}
        self.__cached_fulltext = fulltext
        return fulltext

    def all_tokens(self) -> Iterator[List[ProtobufToken]]:
        """
        Get all tokens of each sentence.
        Returns:
            Each generator generates a sentence which is a list of tokens.
        """
        for sen in self.sentences:
            sentence = []
            for token in sen.token:
                sentence.append(token)
            yield sentence

    def all_parse_trees(self) -> Iterator[ParseTree]:
        """
        Get the parse tree of each sentence.
        Returns:
            Iterator of parse trees of each sentence.
        """
        for sentence in self.sentences:
            yield sentence.parse_tree

    def all_ner(self) -> Iterator[NERMention]:
        """
        Get all ner-mentions in the document.
        Returns:
            Iterator of ner mentions.
        """
        for sentence in self.sentences:
            for mention in sentence.mentions:
                yield mention

    def all_basic_dependencies(self) -> Iterator[DependencyGraph]:
        """
        Get the (basic) dependencies of each sentence in the document.
        (Root first in each dependency graph.)
        Returns:
            Iterator of dependencies of each sentence.
        """
        for sentence in self.sentences:
            yield sentence.basic_dependencies.root_first

    def apply_nlp(self, force: bool = False, **corenlp_settings) -> None:
        """
        Apply nlp-methods to self.text and save it in self.document.
        Args:
            force: Whether to overwrite self.document if it already exists.
            **corenlp_settings: Settings passed to stanfordnlp.server.CoreNLPClient.
                Merged with Document.DEFAULT_NLP_CLIENT_SETTINGS.
        Raises:
            AssertionError: force is False and self.document already exists.
            EmptyParseAttributeException: self.text is empty.
        """
        if not force:
            assert not self.document, "The document is already parsed. Use force=True to overwrite it."
        if not self.fulltext:
            raise EmptyParseAttributeException("The document does not contain any text to apply nlp methods to.")
        corenlp_settings = {**Document.DEFAULT_NLP_CLIENT_SETTINGS, **corenlp_settings}
        # timeout in milliseconds
        with CoreNLPClient(**corenlp_settings, timeout=120*1000) as client:
            self.document = client.annotate(self.fulltext)

    @staticmethod
    def from_text(text: str, apply_nlp: bool = False, **corenlp_settings):  # -> Document
        """
        Create a document from a text-string.
        Args:
            text: The text content of the document.
            apply_nlp: Whether to apply ``self.apply_nlp()`` afterwards.
            **corenlp_settings: Settings passed to stanfordnlp.server.CoreNLPClient.
                Merged with Document.DEFAULT_NLP_CLIENT_SETTINGS.
                (Only used when ``apply_nlp = True``.)

        Returns:
            Document: The document representing the text string.
        """
        document = Document()

        document.text = text
        if apply_nlp:
            document.apply_nlp(**corenlp_settings)

        return document

    @staticmethod
    def from_object(obj: Any,
                    mapping: Optional[Dict[str, str]] = None,
                    apply_nlp: bool = False,
                    **corenlp_settings):
        """
        Parse a document from any arbitrary object.
        Args:
            obj: The object to create a document of.
            mapping: A mapping of attributes, where the document attribute
                maps to the attribute of the object ({doc_attr: obj_attr}).
                Only the items in the mapping will be added to the document.
                Defaults to Document.DEFAULT_MAPPING.
            apply_nlp: Whether to apply ``self.apply_nlp()`` afterwards.
            **corenlp_settings: Settings passed to stanfordnlp.server.CoreNLPClient.
                Merged with Document.DEFAULT_NLP_CLIENT_SETTINGS.
                (Only used when ``apply_nlp = True``.)

        Returns: A document which was parsed from the object.

        """
        if not mapping:
            mapping = Document.DEFAULT_MAPPING
        document = Document()
        document.original = obj

        for obj_attribute_name, document_attribute_name in mapping.items():
            setattr(document, document_attribute_name, getattr(obj, obj_attribute_name))

        if apply_nlp:
            document.apply_nlp(**corenlp_settings)

        return document

    @staticmethod
    def from_news_please(
            article,
            apply_nlp: bool = False,
            **corenlp_settings):  # -> Document
        """
        Parse a document from the news please format.
        Args:
            article: News-please-formatted article.
            apply_nlp: Whether to apply ``self.apply_nlp()`` afterwards.
            **corenlp_settings: Settings passed to stanfordnlp.server.CoreNLPClient.
                Merged with Document.DEFAULT_NLP_CLIENT_SETTINGS.
                (Only used when ``apply_nlp = True``.)
        Returns:
            Document: From news-please format parsed document.
        """
        document = Document()
        document.original = article

        for attribute, value in article.items():
            exists = hasattr(document, attribute) \
                 and getattr(document, attribute) is not None \
                 and not isinstance(getattr(document, attribute), OrderedType) \
                 and attribute not in document.ALLOW_OVERWRITES
            assert not exists
            setattr(document, attribute, value)

        if apply_nlp:
            document.apply_nlp(**corenlp_settings)

        return document
