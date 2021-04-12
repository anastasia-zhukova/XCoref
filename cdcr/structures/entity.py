import shortuuid
import numpy as np
import string
import math
from typing import List, Dict, Union
import copy
import re
import math

from cdcr.entities.dict_lists import LocalDictLists
from cdcr.entities.const_dict_global import *
from cdcr.entities.type_identification import EntityTypeIdentificator
from cdcr.structures.sentiment import Sentiment
from cdcr.structures.candidate import Candidate
from cdcr.structures.document_set import DocumentSet
from cdcr.structures.token import Token

SENTIMENT_PERCENTAGE_THRESHOLD = 0.66
"""Threshold to add a sentiment calculation to the count of calculating the percentage."""


class Entity:
    """
    An Entity class is a placeholder of actors or frequent concepts that have the same meaning or express the same
    concept but have different wording.
    """

    representative_word_num = 5
    ALL = "all"

    def __init__(self, document_set: DocumentSet, members: List[Candidate], name: str = None, ent_preprocessor=None,
                 wikipage=None, last_step=INIT_STEP):
        """

        Args:
            document_set: A collection of documents and their properties
            members: A list of candidates/phrases referring to the same entity
            name: Name of the entity
            ent_preprocessor: A collection of extracted properties from candidates. To create one, use script
                newsalyze/entities/entity_preprocessor.py and create a preprocessor with EntityPreprocessor(document_set,
                entity_class)
            wikipage: A wikipage object from wikipedia page
            last_step: A name of the step/approach at which this entity was created. We strongly recommend to create an
                entity object per candidate_group object contained in document_set.candidates and label entities with
                "init" and then update an entity with resolved mentions. See function add_members.
        """
        self.type_identificator = EntityTypeIdentificator(document_set.configuration.entity_identifier_config.params,
                                                          ent_preprocessor)
        if not len(members):
            raise ValueError("Can't build an entity upon an empty list of members.")

        self.document_set = document_set
        self.members = members
        self.headwords_cand_tree, self.head_tokens = self.calc_headword_tree()
        self.headwords_phrase_tree, self.phrasing_complexity = self.wcl_metric_calc()
        self.wiki_page = wikipage
        self.wiki_page_name = wikipage.title if wikipage is not None else None
        self.type, self.type_details, self.is_single = MISC_TYPE, {}, True
        self.calc_entity_type()
        self.representative_phrases = self.representative_phrases_calc()
        self.id = shortuuid.uuid(name="".join([m.id for m in self.members]))
        self.emotion_inner_frames = []
        self.last_step = last_step
        self.name = name if name is not None else self.get_name()
        self.merge_history = {ORIGINAL_ENTITY: {NAME:  self.name,
                                                REPRESENTATIVE: self.representative,
                                                SIZE: len(self.members),
                                                TYPE: self.type,
                                                PHRASING_COMPLEXITY: float(self.phrasing_complexity),
                                                PHRASES: [(n.text.replace("_", " "), n.id) for n in self.members]}}

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @property
    def sentiment_percentage(self):
        sentiments = {sentiment: 0 for sentiment in Sentiment}
        num = 0
        for member in self.members:
            if member.sentiment.probability > SENTIMENT_PERCENTAGE_THRESHOLD:
                sentiments[member.sentiment.sentiment] += 1
                num += 1
        if num == 0:
            num = 1
        for sentiment in Sentiment:
            sentiments[sentiment] /= num
        return sentiments

    @property
    def representative(self) -> str:
        """
        A name for the entity; the most both frequent and long string amoung all members.

        Returns:
            Entity name

        """
        if type(self.representative_phrases) == list:
            return self.representative_phrases[0]

        # repr_tokens = self.get_repr()[self.ALL]
        repr_tokens = self.representative_phrases[self.ALL][0]
        return "".join([v.word + v.after if i < len(repr_tokens) - 1 else v.word for i, v in enumerate(repr_tokens)])

    @property
    def representatives_doc(self) -> Dict[int, List[Token]]:
        doc_ids = [doc.id for doc in self.document_set]
        return {doc_id: self.representative_phrases[doc_id][0] for doc_id in doc_ids}
        # return self.get_repr(doc_ids)

    @property
    def emotion_similarities(self):
        """

        Returns:
            Identified similarities with other entities.

        """
        return self.emotion_inner_frames

    @emotion_similarities.setter
    def emotion_similarities(self, value):
        self.emotion_inner_frames = value

    def get_repr(self, doc_ids=None) -> Dict[Union[int, str], List[Token]]:
        if doc_ids is None or not len(doc_ids):
            doc_ids = [self.ALL]

        doc_repr = {}

        for doc_id, repr_tokens in self.representative_phrases.items():
            if doc_id not in doc_ids:
                continue
            for repr in repr_tokens:
                if repr[-1].word in self.headwords_cand_tree:
                    if any([w[0].isupper() for w in list(self.headwords_cand_tree)]):
                        if repr[0].word[0].isupper():
                            doc_repr[doc_id] = repr
                            break
                        else:
                            continue
                    else:
                        phrase_dict = {}
                        tokens_dict = {}
                        for m in self.members:
                            if "".join([t.word + t.after for t in repr]) in m.text:
                                phrase_dict[m.text] = phrase_dict.get(m.text, 0) + 1
                                tokens_dict[m.text] = m.tokens
                        phrase_dict = {k:v for k,v in sorted(phrase_dict.items(), reverse=True, key=lambda x: x[1])}
                        if len(phrase_dict):
                            if list(phrase_dict.values())[0] >= self.representative_word_num - 1:
                                doc_repr[doc_id] =  tokens_dict[list(phrase_dict)[0]]
                                break
                        doc_repr[doc_id] = repr
                        break
            if doc_id not in doc_repr:
                words = list({k: v for k, v in sorted(self.headwords_cand_tree.items(), reverse=True,
                                                      key=lambda x: len(x[1]))})
                for w in words:
                    if w in self.head_tokens:
                        doc_repr[doc_id] = [self.head_tokens[w][0]]
                        break
        return doc_repr

    def get_sentiments(self) -> Dict[Sentiment, List[Candidate]]:
        """
        Get all sentiments and their reasoning of this entity.

        Returns:
            The sentiments of this entity and their appearances.
        """
        sentiments = dict([(sentiment, []) for sentiment in list(Sentiment)])
        for member in self.members:
            sentiments[member.sentiment.sentiment].append(member)
        return sentiments

    def add_members(self, members: List[Candidate]):
        """
        Adds mentions from other entities that refer to this entity. Usage: entity1.add_members(entity2.members)

        Args:
            members: Members from another entity.

        """
        self.members.extend(members)

    def update_entity(self, step_name: str = None, update_counter: bool = True, **kwargs):
        """
        Updates entity properties after new members were added.

        Args:
            step_name: Name of a merge step or a criterion based on which the members were added to the entity.
            update_counter: A boolean values indicating to make this status recorded in the entity.
            **kwargs: any custom required parameters required for the update.

        """
        self.update_last_step(step_name, update_counter)
        self.create_name()
        self.update_major_params()
        self.calc_entity_type()
        self.update_merge_history(step_name)
        self.additional_param_update(**kwargs)

    def additional_param_update(self, **kwargs):
        """
        Update custom parameters of entity subclasses.

        Args:
            **kwargs: Parameters requires for the update of custom parameters.

        """
        pass

    def create_name(self):
        """
        Creates a unique entity code that is based on the most representative phrase extracted from entity's members.

        """
        self.representative_phrases_calc()
        self.name = self.get_name()

    def get_name(self) -> str:
        """
        Creates a unique human-readable entity id.

        Returns:
            Entity id-name.

        """
        return self.representative.replace(" ", "_") + "_" + str(len(self.members)) + "_" + \
               re.split(r':', self.last_step)[0] + "_" + str(self.id)[:7]

    def update_major_params(self):
        """
        Updates internal headword-based structures of the entity's members.

        """
        self.headwords_cand_tree, self.head_tokens = self.calc_headword_tree()
        self.headwords_phrase_tree, self.phrasing_complexity = self.wcl_metric_calc()

    def update_merge_history(self, step_name: str):
        """
        Creates a tree-structure merge history, i.e., preserves the order in which members were merged into entities.

        Args:
            step_name: A name of a merge step or a criterion based on which the members were added to the entity.

        """
        if step_name is None:
            return
        if step_name in self.merge_history:
            self.merge_history[step_name].update({
                NAME: self.name,
                REPRESENTATIVE: self.representative,
                SIZE: len(self.members),
                PHRASING_COMPLEXITY: float(self.phrasing_complexity),
                SIM_SCORE: np.mean([ent[SIM_SCORE]
                                    for ent in self.merge_history[step_name][MERGED_ENTITIES]]),
                TYPE: self.type})

    def update_last_step(self, step_name: str, update_counter: bool):
        """
        Update inner entity status of what was the last criterion based on which the recent members were added.

        Args:
            step_name: A name of a merge step or a criterion based on which the members were added to the entity.
            update_counter: A boolean indicaing if one wants to record the last step in the entity.

        """
        if not update_counter or step_name is None:
            return
        self.last_step = step_name

    def calc_entity_type(self):
        """
        Calculates an entity type. See p. 29 of
        https://dke.uni-wuppertal.de/fileadmin/Abteilung/MT/news/MA_thesis_Zhukova.pdf

        """
        if len(self.members):
            self.type, self.type_details = self.type_identificator.calculate_entity_type(self)
        else:
            self.type, self.type_details = MISC_TYPE, {}
        self.is_single = self.type_identificator.is_single({cand.head_token for cand in self.members})

    def calc_headword_tree(self) -> (Dict[str, List[str]], Dict[str, List[Token]]):
        """
        Updates the internal headword-based structure.

        Returns:
            A dictionary with all entity members' ids grouped by similar headword.

        """
        head_dict = {}
        head_tokens = {}
        all_prp = all([m.head_token.pos in ["PRP", "PRP$"] for m in self.members])
        for m in self.members:
            if m.head_token.pos in ["PRP", "PRP$"] and not all_prp:
                continue
            head_dict[m.head_token.word] = [] \
                if m.head_token.word not in head_dict else head_dict[m.head_token.word]
            head_dict[m.head_token.word].append(m.id)
            head_tokens[m.head_token.word] = head_tokens.get(m.head_token.word, []) + [m.head_token]
        return head_dict, head_tokens

    def wcl_metric_calc(self) -> (Dict[str, Dict[str, List[str]]], float):
        """
        Calculates a metric that represents phrasing complexity of an entity, i.e., how various is the wording of the
        phrases referring to this entity. For more details see XCoref paper

        Returns:
            Internal headword-based structure, value of the phrasing complexity

        """
        fractions = []
        sets = []
        cand_dict = {}
        headwords_phrase_tree = {}
        for cand in self.members:
            cand_dict[cand.id] = cand
        for head, cand_ids in self.headwords_cand_tree.items():
            phrase_dict = {}
            for cand_id in cand_ids:
                cand = cand_dict[cand_id]
                word_set = frozenset([w.word for w in cand.tokens
                              if w.word not in string.punctuation and w.word not in LocalDictLists.stopwords])
                phrase_dict[word_set] = phrase_dict.get(word_set, []) + [cand_id]
            # NEW log in the denominator
            fractions.append(len(phrase_dict)/(len(cand_ids)))
            sets.append(len(phrase_dict))
            headwords_phrase_tree[head] = phrase_dict
        score = np.sum(np.array(fractions)) * np.sum(np.array(sets)) / len( self.members) \
                                                                        if len(self.members) > 1 else 1
        return headwords_phrase_tree, float(format(score, '.3f'))

    def representative_phrases_calc(self, lim: int = representative_word_num) -> Dict[str, List[Token]]:
        """
        Extracts the most representative phrases from entity members.

        Args:
            lim: max number of the representative phrases

        Returns:
            A list of representative phrases

        """

        def __sort_by_relevance(phrase_dict):
            return [key for (key, value) in sorted(phrase_dict.items(), reverse=True,
            # key=lambda x: math.log(1 + len(x[0])) * math.log(x[1], 10))]
            key=lambda x: math.log(len(x[0]))  * math.log(len(x[0]) * math.log(1 + x[1], 2)))]

        phrase_doc_dict = {self.ALL: {}}
        phrase_token_dict = {self.ALL: {}}

        for member in self.members:
            phrase_indexes = set().union(*[[dep.governor, dep.dependent] for dep in member.dependency_subtree
                                           if dep.dep in [AMOD, COMPOUND, NMOD_POSS]
                                                and dep.governor_gloss == member.head_token.word])

            tokens = [t for t in member.tokens if len({t.index}.intersection(phrase_indexes))] \
                        if len(phrase_indexes) else [member.head_token]
            phrase = " ".join([t.word for t in tokens])

            if member.document.id not in phrase_doc_dict:
                phrase_doc_dict[member.document.id] = {}
                phrase_token_dict[member.document.id] = {}

            phrase_token_dict[member.document.id][phrase] = tokens
            phrase_token_dict[self.ALL][phrase] = tokens
            phrase_doc_dict[member.document.id][phrase] = phrase_doc_dict[member.document.id].get(phrase, 0) + 1
            phrase_doc_dict[self.ALL][phrase] = phrase_doc_dict[self.ALL].get(phrase, 0) + 1

        if len(phrase_doc_dict[self.ALL]) <= lim:
            self.representative_phrases = {}
            for doc_id, phrase_dict in phrase_doc_dict.items():
                phrases = __sort_by_relevance(phrase_dict)
                self.representative_phrases[doc_id] = [phrase_token_dict[doc_id][p] for p in phrases]
            return self.representative_phrases

        for doc_id, phrase_dict in phrase_doc_dict.items():
            for i, (phrase1, val1) in enumerate(copy.deepcopy(phrase_dict).items()):
                for phrase2, val2 in list(phrase_dict.items())[i + 1:]:
                    sim = []
                    if len(phrase1) >= len(phrase2):
                        main_phrase = phrase1.split(" ")
                        small_phrase = phrase2.split(" ")
                    else:
                        main_phrase = phrase2.split(" ")
                        small_phrase = phrase1.split(" ")
                    m = 0
                    k = 0
                    while k < len(main_phrase):
                        if small_phrase[m] == main_phrase[k]:
                            sim.append(small_phrase[m])
                            m += 1
                        if m == len(small_phrase):
                            k = len(main_phrase)
                        else:
                            k += 1
                    if len(sim) > 0:
                        str_sim = " ".join(sim)
                        phrase_doc_dict[doc_id][str_sim] = phrase_doc_dict[doc_id].get(str_sim, 0) + 1

        self.representative_phrases = {}
        for doc_id, phrase_dict in phrase_doc_dict.items():
            selected_list = __sort_by_relevance(phrase_dict)
            self.representative_phrases[doc_id] = [phrase_token_dict[doc_id][p] for p in selected_list
                                                   if p in phrase_token_dict[doc_id]][:min(len(selected_list),
                                                                                         self.representative_word_num)]
        return self.representative_phrases
