from cdcr.entities.dict_lists import LocalDictLists
import cdcr.config as config
from cdcr.entities.const_dict_global import *
from cdcr.logger import LOGGER
from cdcr.structures.document_set import DocumentSet
from cdcr.structures.token import Token

import string
import re
import progressbar

MESSAGES = {
    "preproccess": "PROGRESS: Creating an entity out of the %(value)d-th/ %(max_value)d-th (%(percentage)d %%) candidate group "
                   "(in: %(elapsed)s).",
    "key": "The key \"{}\" already exists. The new entity won't be added. "
}


class EntityPreprocessor:
    """
    A collection of extracted properties from candidates, e.g., dictionaries with NER types.
    """

    logger = LOGGER

    def __init__(self, docs: DocumentSet, entity_class):
        """

        Args:
            docs: A collection of documents and their properties.
            entity_class: A class of entities that you want to convert candidate groups into entities.
        """

        self.docs = docs
        self.entity_class = entity_class

        self.config_params = docs.configuration.entity_identifier_config.params

        self.cand_dict = {}
        self.ner_counted_dict = {PERSON_NER: {}, ORGANIZATION_NER: {}, COUNTRY_NER: {}, NATIONALITY_NER: {},
                                 STATE_NER: {}, LOCATION_NER: {}, CITY_NER: {}, IDEOLOGY_NER: {}, MISC_NER: {},
                                 TITLE_NER: {}}

        self.labeling_dict = {}
        self.compound_dict = {}
        self.ner_dict = {}
        self.phrase_ner_dict = {}

        # supportive dictionaries
        self._ner_list_construction()

    @staticmethod
    def not_stopword(word):
        return word.lower() not in LocalDictLists.stopwords and word.lower() not in string.punctuation

    def _ner_list_construction(self):
        """
        Takes all candidates and creates NER dictionaries structures in different ways. Resolves some conflicts when
        the same phrases were annotated differently across the analyzed text.

        """
        token_ner_dict = {}
        for doc in self.docs:
            for sent in doc.all_tokens():
                for token in sent:
                    if token.word not in token_ner_dict:
                        token_ner_dict[token.word] = {}
                    token_ner_dict[token.word][token.ner] = \
                                            token_ner_dict[token.word].get(token.ner, 0) + 1
            for ner in doc.all_ner():
                if ner.ner in self.ner_counted_dict:
                    if ner.text.lower() not in LocalDictLists.pronouns:
                        text_array = ner.text.split(" ")
                        text = "_".join(text_array)
                        self.ner_counted_dict[ner.ner][text] = self.ner_counted_dict[ner.ner].get(text, 0) + 1
        token_ner_dict = {key: {k:v for k, v in sorted(ners.items(),reverse=True, key=lambda x: x[1])}
                                    for key, ners in token_ner_dict.items()}

        for word, ners in token_ner_dict.items():
            if list(ners.keys())[0] != NON_NER:
                try:
                    self.ner_counted_dict[list(ners.keys())[0]][word] = list(ners.values())[0]
                except KeyError:
                    continue

        ner_overlap_1 = set(list(self.ner_counted_dict[PERSON_NER].keys())) & \
                        set(list(self.ner_counted_dict[ORGANIZATION_NER].keys()))
        ner_overlap_2 = set(list(self.ner_counted_dict[PERSON_NER].keys())) & \
                        set(list(self.ner_counted_dict[LOCATION_NER].keys()))

        # cleanup of the NE misclassification
        if len(ner_overlap_1) > 0:
            for ner in list(ner_overlap_1):
                if self.ner_counted_dict[ORGANIZATION_NER][ner] >= self.ner_counted_dict[PERSON_NER][ner]:
                    self.ner_counted_dict[PERSON_NER].pop(ner)
                else:
                    self.ner_counted_dict[ORGANIZATION_NER].pop(ner)

        if len(ner_overlap_2) > 0:
            for ner in list(ner_overlap_2):
                if ner in self.ner_counted_dict[PERSON_NER] and ner in self.ner_counted_dict[LOCATION_NER]:
                    if self.ner_counted_dict[PERSON_NER][ner] >= self.ner_counted_dict[LOCATION_NER][ner]:
                        self.ner_counted_dict[LOCATION_NER].pop(ner)
                    else:
                        self.ner_counted_dict[PERSON_NER].pop(ner)

        # conversion from counts to lists
        for type, ner in self.ner_counted_dict.items():
            self.ner_dict[type] = set(list(ner.keys()))

        to_remove = []
        for ner in self.ner_dict[PERSON_NER]:
            if ner[:-1] in self.ner_dict[ORGANIZATION_NER]:
                self.ner_dict[ORGANIZATION_NER].add(ner)
                to_remove.append(ner)

        for ner in to_remove:
            self.ner_dict[PERSON_NER].remove(ner)

        self.ner_dict[COUNTRY_NER] = self.ner_dict[COUNTRY_NER].union(self.ner_dict[STATE_NER])\
                                        .union(self.ner_dict[LOCATION_NER]).union(self.ner_dict[CITY_NER])
        self.ner_dict.pop(STATE_NER)
        self.ner_dict.pop(LOCATION_NER)
        self.ner_dict.pop(CITY_NER)

        to_remove = []
        for ner in self.ner_dict[MISC_NER]:
            if ner[:-1] in self.ner_dict[IDEOLOGY_NER]:
                self.ner_dict[IDEOLOGY_NER].add(ner)
                to_remove.append(ner)
                continue
            if ner + "s" in self.ner_dict[IDEOLOGY_NER]:
                self.ner_dict[IDEOLOGY_NER].add(ner)
                to_remove.append(ner)
                continue
            if ner[:-1] in self.ner_dict[NATIONALITY_NER]:
                self.ner_dict[NATIONALITY_NER].add(ner)
                to_remove.append(ner)
                continue
            if ner[-2:] == "ns":
                self.ner_dict[NATIONALITY_NER].add(ner)
                to_remove.append(ner)
                continue

        self.ner_dict[ORGANIZATION_NER] = self.ner_dict[ORGANIZATION_NER].union(self.ner_dict[IDEOLOGY_NER])
        self.ner_dict[ORGANIZATION_NER] = self.ner_dict[ORGANIZATION_NER] - self.ner_dict[TITLE_NER]
        self.ner_dict[COUNTRY_NER] = self.ner_dict[COUNTRY_NER] - self.ner_dict[ORGANIZATION_NER]
        self.ner_dict.pop(IDEOLOGY_NER)
        self.ner_dict.pop(MISC_NER)
        self.ner_dict.pop(TITLE_NER)

        for ner_type, phrases in self.ner_dict.items():
            for phrase in phrases:
                split = " ".join([w for w in re.split(r'(-|_)', phrase) if w not in string.punctuation])
                self.phrase_ner_dict[split] = ner_type

    def _ner_check(self, token: Token) -> str:
        """
        Checks is a token is NER, if not returns a word in lowercase

        Args:
            token: A token under consideration.

        Returns:
            A lowercase word if necessary.
        """
        all_ner = set()
        for n in list(self.ner_counted_dict.values()):
            all_ner.update(list(n.keys()))
        if token.word in all_ner or token.lemma in all_ner or token.word[:-1] in all_ner:
            return token.word
        if token.ner != NON_NER:
            return token.word
        if token.word in LocalDictLists.titles:
            return token.word
        return token.word.lower()

    def _leave_cand(self, cand) -> bool:
        """
        Filter out some mentions
        Args:
            cand: a candidate mention

        Returns: a flag is a candidate is filtered out or not.

        """
        # if extracted and not annotated, perform cand post processing
        if cand.annot_text == "" or cand.annot_text is None:

            # ignore NPs with only one article-word
            if len(cand.tokens) == 1 and cand.tokens[0].pos == DT:
                return False

            # ignore one-word NPs consisting of single adjectives
            if (JJ in cand.head_token.pos and cand.head_token.ner == NON_NER and
                len([t.word for t in cand.tokens if self.not_stopword(t.word)]) == 1) \
                    and self.config_params.preprocessing.exclude_single_adj:
                return False

            # ignore NPs where a head word is a general phrase, e.g., everything, nothing, etc
            if cand.head_token.word in LocalDictLists.general_nouns and \
                    self.config_params.preprocessing.exclude_general_nouns:
                return False

            # ignore single-word NPs with titles Mr, Ms, etc.
            if cand.head_token.word in LocalDictLists.titles:
                return False

            # ignore candidates related to time, date, and duration
            if cand.head_token.ner in [TIME_NER, DATE_NER, DURATION_NER] and \
                    self.config_params.preprocessing.exclude_time:
                return False

            # ignore NPs where a root word is a words like "those", "some", etc.
            # if cand.head_token.pos in [DT, POS] and self.config_params.preprocessing.exclude_dt:
            #     return None

            # ignore pronominals from coref groups
            # if cand.coref_subtype == PRONOMINAL:
            #     return None
        return True

    def entity_dict_construction(self, **kwargs):
        """
        Converts a list of candidate groups into dictionary of entities

        Args:
            **kwargs: parameters required for custom entity_dict creation.

        Returns:
            Dict[str, Entity] : a dictionary of entities.
        """
        entity_dict = {}
        widgets = [
            progressbar.FormatLabel(MESSAGES["preproccess"])
        ]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      maxval=len(self.docs.candidates)).start()

        for i, cand_group in enumerate(sorted(self.docs.candidates, reverse=True, key=lambda x: len(x))):
            cand_selected = []

            for cand in cand_group:

                if not self._leave_cand(cand):
                    continue

                cand_selected.append(cand)

            if not len(cand_selected):
                continue

            ent = self.entity_class(document_set=self.docs, members=cand_selected, name=None, ent_preprocessor=self,
                                    wikipage=None)
            entity_dict[ent.name] = ent
            bar.update(i+1)
        bar.finish()

        return entity_dict
