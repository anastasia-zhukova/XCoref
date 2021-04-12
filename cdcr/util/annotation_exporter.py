import pickle
import string
from contextlib import suppress
from typing import List

from cdcr.structures import DocumentSet
from structures import Document
from structures.candidate import Candidate
from structures.entity import Entity
from structures.sentence import Sentence
import os
import re


DEFAULT_DIRECTORY = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "exported_annotations")
)
DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"
FILE_ENDING = ".pickle"  # must start with a .
ANNOTATIONS_SUFFIX = ".annotations"
DOCUMENT_SET_SUFFIX = ".document_set"


class AllsidesAnnotationExample:
    id: str

    article_bias: str
    article_source: str

    row_index: int

    sentence_index: int

    sentence: str
    sentence_in_doc_from: int
    sentence_in_doc_to: int

    sentence_normalized: str
    sentence_normalized_in_doc_from: int
    sentence_normalized_in_doc_to: int

    story_id: int

    # Target main mention: The best describing mention of this entity (i.e. target) in one document.
    target_main_mention_in_doc: str
    target_main_mention_in_doc_from: int
    target_main_mention_in_doc_to: int

    target_main_mention_normalized_in_doc: str
    target_main_mention_normalized_in_doc_from: int
    target_main_mention_normalized_in_doc_to: int

    # Target mention: The current mention.
    target_mention: str
    target_mention_in_doc_from: int
    target_mention_in_doc_to: int
    target_mention_in_sentence_from: int
    target_mention_in_sentence_to: int
    target_mention_in_sentence_normalized_from: int
    target_mention_in_sentence_normalized_to: int

    target_mention_normalized: str
    target_mention_normalized_in_doc_from: int
    target_mention_normalized_in_doc_to: int
    target_mention_normalized_in_sentence_from: int
    target_mention_normalized_in_sentence_to: int
    target_mention_normalized_in_sentence_normalized_from: int
    target_mention_normalized_in_sentence_normalized_to: int

    # Primary mention: Mention of the most important entity in the same sentence.
    # THIS MUST NOT BE THE SAME ENTITY AS WHICH self.target_mention DESCRIBES, BUT MAY BE!
    primary_mention: str
    primary_mention_in_doc_from: int
    primary_mention_in_doc_to: int
    primary_mention_in_sentence_from: int
    primary_mention_in_sentence_to: int
    primary_mention_in_sentence_normalized_from: int
    primary_mention_in_sentence_normalized_to: int

    primary_mention_normalized: str
    primary_mention_normalized_in_doc_from: int
    primary_mention_normalized_in_doc_to: int
    primary_mention_normalized_in_sentence_from: int
    primary_mention_normalized_in_sentence_to: int
    primary_mention_normalized_in_sentence_normalized_from: int
    primary_mention_normalized_in_sentence_normalized_to: int

    # Preferred mention: Best mention of the target in the same sentence.
    preferred_mention: str
    preferred_mention_in_doc_from: int
    preferred_mention_in_doc_to: int
    preferred_mention_in_sentence_from: int
    preferred_mention_in_sentence_to: int
    preferred_mention_in_sentence_normalized_from: int
    preferred_mention_in_sentence_normalized_to: int

    preferred_mention_normalized: str
    preferred_mention_normalized_in_doc_from: int
    preferred_mention_normalized_in_doc_to: int
    preferred_mention_normalized_in_sentence_from: int
    preferred_mention_normalized_in_sentence_to: int
    preferred_mention_normalized_in_sentence_normalized_from: int
    preferred_mention_normalized_in_sentence_normalized_to: int

    is_preferred: bool

    input_for_absa: list
    date_publish: str

    multiple_appearances_in_sentence: bool


class AnnotationExporter:
    """
    AnnotationExporter parses a DocumentSet into AllsidesAnnotationExample and returns them via parse().
    """

    MIN_MENTIONS_PER_SENTENCE = 0.02
    """float: Minimum number of mentions per sentence over all documents."""

    def __init__(self, document_set: DocumentSet):
        super().__init__()
        self.document_set: DocumentSet = document_set
        self.entity_mentions = None
        self.entities = None
        self.appearances = None
        self.normalized_data = None
        self.num_document_sentences = sum(
            len(document.sentences) for document in self.document_set
        )
        self.primaries = None
        self.filename = None
        self.topic = None

    def preprocess(self):
        self.entity_mentions = {}
        self.entities = {
            entity.id: entity
            for entity in self.document_set.entities
            if self.is_entity_relevant(entity)
        }
        self.normalized_data = {}

        self.appearances = self._appearances()
        self.primaries = self._primaries()

        self.topic = re.sub("[^0-9a-zA-Z_-]+", "-", self.document_set.topic)
        self.filename = (
            self.document_set.processing_information.last_module_time.strftime(
                DATE_FORMAT
            )
            + "_"
            + self.topic
        )

    def _primaries(self):
        primaries = {}
        a = self.appearances
        for document in a:
            primaries[document] = {}
            for sentence in a[document]:
                primary_entity = self._primary_entity(list(a[document][sentence].keys()))
                primaries[document][sentence] = self._preferred_mention(a[document][sentence][primary_entity])
        return primaries

    def _appearances(self):
        appearances = {
            doc: {
                sent: {ent: [] for ent in self.entities.values()}
                for sent in doc.sentences
            }
            for doc in self.document_set
        }
        for entity in self.entities.values():
            for member in entity.members:
                appearances[member.document][member.sentence][entity].append(
                    member
                )
        for document in appearances:
            for sentence in appearances[document]:
                self._remove_empty(appearances[document][sentence])
            self._remove_empty(appearances[document])
        self._remove_empty(appearances)
        return appearances

    @staticmethod
    def _remove_empty(dict_):
        for key in [key for key in dict_ if len(dict_[key]) == 0]:
            del dict_[key]

    @staticmethod
    def _preferred_mention(mentions):
        assert len(mentions) > 0, "mentions is empty."
        if len(mentions) == 1:
            return mentions[0]
        scores = {mention: 0 for mention in mentions}
        first_mention = True
        for mention in mentions:
            if first_mention:
                scores[mention] += 1
                first_mention = False
            if mention.is_representative:
                scores[mention] += 3
            if mention.coref_subtype.startswith("PROP"):
                scores[mention] += 2
            if len(mention.tokens) == 2:  # i.e. title+name or first+last name
                scores[mention] += 2
        return max(scores, key=lambda k: scores[k])

    @staticmethod
    def _primary_entity(entities):
        primary = entities[0]
        for e in entities:
            if len(e.members) > len(primary.members):
                primary = e
        return primary

    def is_entity_relevant(self, entity):
        return len(
            entity.members
        ) / self.num_document_sentences >= self.MIN_MENTIONS_PER_SENTENCE and entity.type.lower().startswith(
            "person"
        )

    def has_entity_multiple_appearances(self, document, sentence, entity):
        return len(self.appearances[document][sentence][entity]) > 1

    # def parse(self) -> List[AllsidesAnnotationExample]:
    def parse(self) -> List[dict]:
        annotations = []

        for document, sentences in self.appearances.items():
            for sentence, entities in sentences.items():
                for entity, members in entities.items():
                    for member in members:
                        annotations.append(
                            self._parse_annotation(member, entity, sentence, document)
                        )

        annotations_as_dicts = []
        for annotation in annotations:
            annotations_as_dicts.append(annotation.__dict__)

        # return annotations
        return annotations_as_dicts

    def export(self, annotations_location: str = None, document_set_location: str = None):
        if not annotations_location:
            annotations_location = os.path.join(DEFAULT_DIRECTORY, self.filename + ANNOTATIONS_SUFFIX + FILE_ENDING)
        if not document_set_location:
            document_set_location = os.path.join(DEFAULT_DIRECTORY, self.filename + DOCUMENT_SET_SUFFIX + FILE_ENDING)
        pickle.dump(
            self.parse(),
            open(annotations_location, "wb"),
        )
        pickle.dump(
            self.document_set,
            open(document_set_location, "wb"),
        )

    def _normalize(self, text, begin_char, end_char):
        with suppress(KeyError):
            return self.normalized_data[begin_char][end_char][text]
        begin_char_additions = 0
        end_char_subtractions = 0

        while begin_char_additions < len(text):
            if text[begin_char_additions] in string.whitespace:
                begin_char_additions += 1
            else:
                break
        while end_char_subtractions < len(text) - begin_char_additions:
            if text[(end_char_subtractions * -1) - 1] in string.whitespace:
                end_char_subtractions += 1
            else:
                break

        if begin_char not in self.normalized_data:
            self.normalized_data[begin_char] = {}
        if end_char not in self.normalized_data[begin_char]:
            self.normalized_data[begin_char][end_char] = {}

        new_text = text[
            begin_char_additions : end_char_subtractions * -1
            if end_char_subtractions
            else len(text)
        ]
        self.normalized_data[begin_char][end_char][text] = (
            new_text,
            begin_char,
            end_char,
            begin_char_additions,
            end_char_subtractions,
        )

        return self.normalized_data[begin_char][end_char][text]

    def _parse_annotation(
        self, member: Candidate, entity: Entity, sentence: Sentence, document: Document
    ):
        a = AllsidesAnnotationExample()
        target_main = entity.representatives_doc[document.id]
        primary = self.primaries[document][sentence]
        preferred = self._preferred_mention(self.appearances[document][sentence][entity])

        a.article_bias = None

        try:
            a.article_source = document.url
        except AttributeError:
            a.article_source = None

        a.row_index = document.id

        a.sentence_index = sentence.index

        a.sentence = sentence.text
        a.sentence_in_doc_from = sentence.begin_char
        a.sentence_in_doc_to = sentence.end_char

        (
            a.sentence_normalized,
            a.sentence_normalized_in_doc_from,
            a.sentence_normalized_in_doc_to,
            norm_sent_add,
            norm_sent_sub,
        ) = self._normalize(sentence.text, sentence.begin_char, sentence.end_char)

        a.story_id = None

        # Target main mention: The best describing mention of this entity (i.e. target) in one document.
        a.target_main_mention_in_doc = document.fulltext[
            target_main[0].beginChar: target_main[-1].endChar
        ]
        a.target_main_mention_in_doc_from = target_main[0].beginChar
        a.target_main_mention_in_doc_to = target_main[-1].endChar

        (
            a.target_main_mention_normalized_in_doc,
            a.target_main_mention_normalized_in_doc_from,
            a.target_main_mention_normalized_in_doc_to,
            _,
            _,
        ) = self._normalize(
            a.target_main_mention_in_doc,
            a.target_main_mention_in_doc_from,
            a.target_main_mention_in_doc_to,
        )

        # Target mention: The current mention.
        a.target_mention = member.text
        a.target_mention_in_doc_from = member.tokens[0].beginChar
        a.target_mention_in_doc_to = member.tokens[-1].endChar
        a.target_mention_in_sentence_from = member.tokens[0].sentence_begin_char
        a.target_mention_in_sentence_to = member.tokens[-1].sentence_end_char
        a.target_mention_in_sentence_normalized_from = member.tokens[0].sentence_begin_char - norm_sent_add
        a.target_mention_in_sentence_normalized_to = member.tokens[-1].sentence_end_char - norm_sent_add

        (
            a.target_mention_normalized,
            a.target_mention_normalized_in_doc_from,
            a.target_mention_normalized_in_doc_to,
            norm_ment_add,
            norm_ment_sub,
        ) = self._normalize(
            member.text, member.tokens[0].beginChar, member.tokens[-1].endChar
        )

        a.target_mention_normalized_in_sentence_from = member.tokens[0].sentence_begin_char + norm_ment_add
        a.target_mention_normalized_in_sentence_to = member.tokens[-1].sentence_end_char - norm_ment_sub
        a.target_mention_normalized_in_sentence_normalized_from = member.tokens[0].sentence_begin_char - norm_sent_add + norm_ment_add
        a.target_mention_normalized_in_sentence_normalized_to = member.tokens[-1].sentence_end_char - norm_sent_add - norm_ment_sub

        # Primary mention: Mention of the most important entity in the same sentence.
        # THIS MUST NOT BE THE SAME ENTITY AS WHICH self.target_mention DESCRIBES, BUT MAY BE!
        a.primary_mention = primary.text
        a.primary_mention_in_doc_from = primary.tokens[0].beginChar
        a.primary_mention_in_doc_to = primary.tokens[-1].endChar
        a.primary_mention_in_sentence_from = primary.tokens[0].sentence_begin_char
        a.primary_mention_in_sentence_to = primary.tokens[-1].sentence_end_char
        a.primary_mention_in_sentence_normalized_from = primary.tokens[0].sentence_begin_char - norm_sent_add
        a.primary_mention_in_sentence_normalized_to = primary.tokens[-1].sentence_end_char - norm_sent_add

        (
            a.primary_mention_normalized,
            a.primary_mention_normalized_in_doc_from,
            a.primary_mention_normalized_in_doc_to,
            norm_prim_add,
            norm_prim_sub
        ) = self._normalize(
            primary.text,
            primary.tokens[0].beginChar,
            primary.tokens[-1].endChar
        )
        a.primary_mention_normalized_in_sentence_from = primary.tokens[0].sentence_begin_char + norm_prim_add
        a.primary_mention_normalized_in_sentence_to = primary.tokens[-1].sentence_end_char - norm_prim_sub
        a.primary_mention_normalized_in_sentence_normalized_from = primary.tokens[0].sentence_begin_char - norm_sent_add + norm_prim_add
        a.primary_mention_normalized_in_sentence_normalized_to = primary.tokens[-1].sentence_end_char - norm_sent_add - norm_prim_sub

        # Preferred mention: Best mention of the target in the same sentence.
        a.preferred_mention = preferred.text
        a.preferred_mention_in_doc_from = preferred.tokens[0].beginChar
        a.preferred_mention_in_doc_to = preferred.tokens[-1].endChar
        a.preferred_mention_in_sentence_from = preferred.tokens[0].sentence_begin_char
        a.preferred_mention_in_sentence_to = preferred.tokens[-1].sentence_end_char
        a.preferred_mention_in_sentence_normalized_from = preferred.tokens[0].sentence_begin_char - norm_sent_add
        a.preferred_mention_in_sentence_normalized_to = preferred.tokens[-1].sentence_end_char - norm_sent_add

        (
            a.preferred_mention_normalized,
            a.preferred_mention_normalized_in_doc_from,
            a.preferred_mention_normalized_in_doc_to,
            norm_pref_add,
            norm_pref_sub
        ) = self._normalize(
            preferred.text,
            preferred.tokens[0].beginChar,
            preferred.tokens[-1].endChar
        )
        a.preferred_mention_normalized_in_sentence_from = preferred.tokens[0].sentence_begin_char + norm_pref_add
        a.preferred_mention_normalized_in_sentence_to = preferred.tokens[-1].sentence_end_char - norm_pref_sub
        a.preferred_mention_normalized_in_sentence_normalized_from = preferred.tokens[0].sentence_begin_char - norm_sent_add + norm_pref_add
        a.preferred_mention_normalized_in_sentence_normalized_to = preferred.tokens[-1].sentence_end_char - norm_sent_add - norm_pref_sub

        a.is_preferred = preferred == member

        #
        a.input_for_absa = None
        try:
            a.date_publish = document.date_publish
        except AttributeError:
            a.date_publish = None

        a.multiple_appearances_in_sentence = self.has_entity_multiple_appearances(
            document, sentence, entity
        )

        a.id = f"{self.topic}_{a.row_index}_{a.sentence_index}_{a.target_main_mention_normalized_in_doc}_{a.target_mention_in_sentence_from}_{a.target_mention_in_sentence_to}"
        return a

    @staticmethod
    def run(document_set: DocumentSet) -> DocumentSet:
        exporter = AnnotationExporter(document_set)
        exporter.preprocess()
        exporter.export()
        return document_set
