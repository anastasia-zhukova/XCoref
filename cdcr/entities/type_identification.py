from cdcr.entities.const_dict_global import *
from cdcr.structures.token import Token
from cdcr.structures.params import Params
from cdcr.entities.entity_preprocessor import EntityPreprocessor
# from newsalyze.structures.entity import Entity

import numpy as np
import copy
from nltk.corpus import wordnet as wn

from typing import *


ENTITY_TYPES_MATCH = {
    PERSON_TYPE: {NON_NE_TYPE: PERSON_WN,
                  NE_TYPE: PERSON_NER},
    GROUP_TYPE: {NON_NE_TYPE: GROUP_WN,
                 NE_TYPE: ORGANIZATION_NER},
    COUNTRY_TYPE: {NON_NE_TYPE: LOCATION_WN,
                   NE_TYPE: COUNTRY_NER}
}

ENTITY_TYPE_PRIORITY = {
    PERSON_TYPE: 2,
    GROUP_TYPE: 1,
    COUNTRY_TYPE: 0
}


class EntityTypeIdentificator:
    """
    A class that contains methods to identify entity's class, e.g., person, country, group of people, etc.
    Identification is based on NER labels and WordNet lexnames.
    See details on p.29 in https://dke.uni-wuppertal.de/fileadmin/Abteilung/MT/news/MA_thesis_Zhukova.pdf
    """

    def __init__(self, config_params: Params, preprocessor: EntityPreprocessor = None):
        """

        Args:
            config_params: A config that contains information about entity preprocessing. Config values are defined in
                ParamsEntities class.)
            preprocessor: A class that contain different properties from candidates to be merged. See entity.py
                for more details.
        """

        self.ent_preprocessor = preprocessor

        self.infreq_entity_type_threshold = 0.25
        if hasattr(config_params, "preprocessing"):
            if hasattr(config_params.preprocessing, "infreq_entity_type_threshold"):
                self.infreq_entity_type_threshold = config_params.preprocessing.infreq_entity_type_threshold

        self._entity_types_function = {
            NON_NE_TYPE: self._is_general_category,
            NE_TYPE: self._is_ner_category
        }

    def calculate_entity_type(self, entity) -> (str, Dict[Any, Any]):
        """
        Identifies an entity type.

        Args:
            entity: An Entity object or its subclasses.

        Returns:
            An entity type and a structure based on which the identification was done.

        """
        tree = entity.headwords_cand_tree
        # for app in (entity.appos_dict):
        #     if app not in tree:
        #         tree[app] = [m.id for m in entity.members if app in m.text]
        type_calc_dict = self._type_dict_init(tree)

        word_proc = {w: False for w in list(tree.keys())}
        i = 0
        for head_token in {cand.head_token for cand in entity.members}:
            if head_token.word not in word_proc:
                continue
            if word_proc[head_token.word]:
                continue
            word_proc[head_token.word] = True
            for ent_type, ner_types in ENTITY_TYPES_MATCH.items():

                for ner_type, ner_code in ner_types.items():
                    type_calc_dict[ent_type][ner_type][i] = len(entity.headwords_cand_tree[head_token.word]) * float(
                        self._entity_types_function[ner_type](head_token, ner_code))
            i += 1

        return self._type_dict_sum(type_calc_dict, entity)

    def _type_dict_init(self, head_words: Dict[str, Any]) -> Dict[Any, Any]:
        """
        Forms a dictionary to be filled with values based on which the entity type will be identified.

        Args:
            head_words: All headwords extracted from the entity's members.

        Returns:
            An initialized dictionary for values.

        """
        type_calc_dict = copy.deepcopy(ENTITY_TYPES_MATCH)
        for ent_type, ner_types in type_calc_dict.items():
            for ner_type, ner_code in ner_types.items():
                type_calc_dict[ent_type][ner_type] = [0] * len(head_words)
        return type_calc_dict

    def _type_dict_sum(self, type_calc_dict: Dict[Any, Any], entity) -> (str, Dict[Any, Any]):
        """
        Identifies a priority entity type.

        Args:
            type_calc_dict: An initialized dictionary for values.
            entity:

        Returns:
            An entity type and a structure based on which the identification was done.

        """
        is_single = self.is_single({cand.head_token for cand in entity.members})
        type_calc_dict_filtered = {ent_type: ner_types for ent_type, ner_types in type_calc_dict.items()
                                   if np.sum(np.array(list(ner_types.values()))) > 0}

        type_calc_dict_filtered = {ent_type: ner_types for ent_type, ner_types in
                                   sorted(type_calc_dict_filtered.items(),
                                          reverse=True, key=lambda x:
                                       (np.sum(np.array(list(x[1].values()))), ENTITY_TYPE_PRIORITY[x[0]]))}

        if len(type_calc_dict_filtered) == 0:
            wiki = {entity.wiki_page_name} if entity.wiki_page_name is not None else set()
            if any([v in self.ent_preprocessor.phrase_ner_dict
                                for v in set([m.text.replace("_", " ") for m in entity.members]).union(wiki)] ):
                return GROUP_TYPE + "-" + NE_TYPE, type_calc_dict
            return MISC_TYPE, type_calc_dict

        else:
            if PERSON_TYPE in type_calc_dict_filtered and GROUP_TYPE in type_calc_dict_filtered \
                    and len(type_calc_dict_filtered) >= 2:

                if np.sum(np.array(type_calc_dict_filtered[GROUP_TYPE][NE_TYPE])) \
                        > np.sum(np.array(list(type_calc_dict_filtered[PERSON_TYPE].values()))):
                    return GROUP_TYPE + "-" + NE_TYPE, type_calc_dict
                elif np.sum(np.array(type_calc_dict_filtered[GROUP_TYPE][NE_TYPE])) > 0:
                    if not is_single:
                        return PERSON_TYPE + "-" + NES_TYPE, type_calc_dict

            selected_ent_type, selected_ner_types = list(type_calc_dict_filtered.items())[0]
            if np.sum(np.array(selected_ner_types[NE_TYPE])) > 0 and selected_ent_type != GROUP_TYPE:
                    return selected_ent_type + "-" + NE_TYPE, type_calc_dict
            elif np.sum(np.array(selected_ner_types[NE_TYPE])) / np.sum(np.array(selected_ner_types[NON_NE_TYPE])) >= 0.15 \
                and selected_ent_type == GROUP_TYPE:
                return selected_ent_type + "-" + NE_TYPE, type_calc_dict

            if selected_ent_type == PERSON_TYPE:
                if is_single:
                    return selected_ent_type + "-" + NN_TYPE, type_calc_dict
                else:
                    return selected_ent_type + "-" + NNS_TYPE, type_calc_dict

            # if selected_ent_type == GROUP_TYPE and entity.wiki_page_name is not None and \
            #                         any([w.capitalize() in self.ent_preprocessor.phrase_ner_dict
            #                         for w in list( entity.headwords_cand_tree.keys())]):
            #     return GROUP_TYPE + "-" + NE_TYPE, type_calc_dict

            return selected_ent_type, type_calc_dict

    def _is_general_category(self, token: Token, wn_type: str) -> int:
        """
        Calculates a value of a weight and frequency of a provided lexical name among all word's WordNet synsets.

        Args:
            token: A headword token.
            wn_type: A WordNet lexname under consideration.

        Returns:
            A weight/influence-value of how much a word represents a provided lexname.

        """
        coeff = 0
        count = 0
        word_modif = token.word.replace("-", "")

        try:
            syn = wn.synsets(token.word) if len(wn.synsets(token.word)) >= len(wn.synsets(word_modif)) \
                                            else wn.synsets(word_modif)
        except RecursionError:
            if len(token.word.split("-")) > 1:
                try:
                    syn = wn.synsets(token.word.split("-")[-1])
                except RecursionError:
                    return 0
            else:
                syn = []

        if not len(syn) and len(token.word.split("-")) > 1:
            try:
                syn = wn.synsets(token.word.split("-")[-1])
            except RecursionError:
                return 0

        if not len(syn) and len(token.word) > 2 and wn_type == PERSON_WN:
            # wordnet doesn't always contain two types of writing, e.g., "protester" is in the dict but not "protestor"
            if token.word[-3:] == "ors":
                word_modif = token.word[:-3] + "ers"
            if token.word[-2:] == "or":
                word_modif = token.word[:-2] + "er"

            try:
                syn = wn.synsets(word_modif)
            except RecursionError:
                return 0

        for i_id, s in enumerate(syn):
            if s.lexname() == wn_type:
                if wn_type == GROUP_WN:
                    if "people" in s.definition() or token.word.lower() == "people"  \
                            or "institution" in s.definition() \
                            or "social" in s.definition() \
                            or "person" in s.definition():
                        count += 1
                        coeff += (2 * len(syn) - 2 * i_id) / len(syn)
                    else:
                        # in other cases, add smaller coeff and increase counter partially
                        count += 0.8
                        coeff += (len(syn) - i_id) / len(syn)
                        break
                else:
                    count += 1
                    coeff += (2 * len(syn) - 2 * i_id) / len(syn)

        if len(syn) > 0:
            avg_coeff = coeff * count / len(syn)
            if "group" in token.word.lower() and wn_type == GROUP_WN:
                avg_coeff = self.infreq_entity_type_threshold
            if avg_coeff >= self.infreq_entity_type_threshold:
                return avg_coeff
            return 0

        # wordnet doesn't always contain nouns originated from adjectives, e.g., "illegals"
        elif len(wn.synsets(token.word[:-1])) > 0 and token.word[-1] == "s":
            if "NN" in token.pos and wn_type == PERSON_WN:
                return self.infreq_entity_type_threshold
        return 0

    def _is_ner_category(self, token: Token, ner_type: str) -> int:
        """
        Calculates how many times a token has a provided NER label.

        Args:
            token: A headword token.
            ner_type:

        Returns:

        """
        if token.word in self.ent_preprocessor.ner_dict[ner_type]:
            return True
        return False

    def is_single(self, head_tokens: Union[List, Set]) -> bool:
        """
        Identifies if an entity is single or plural.

        Args:
            head_tokens: A collection of headword tokens.

        Returns:
            If an entity is single.

        """
        nn = np.zeros(len(head_tokens))
        nns = np.zeros(len(head_tokens))
        for i, head_token in enumerate(head_tokens):
            if head_token.pos in [NN_TYPE.upper(), "NNP"]:
                nn[i] = 1
            if head_token.pos in [NNS_TYPE.upper(), "NNPS"] \
                    or head_token in self.ent_preprocessor.ner_dict[NATIONALITY_NER]:
                nns[i] = 1
        if np.sum(nns) >= np.sum(nn) and np.sum(nns) > 0:
            return False
        else:
            return True
