from cdcr.entities.const_dict_global import *
from cdcr.config import *
from cdcr.entities.dict_lists import LocalDictLists
from cdcr.structures.entity import Entity
from cdcr.structures.token import Token
from typing import *
from cdcr.entities.sieve_based.params_sieves import SievesParams
from cdcr.entities.sieve_based.entity_preprocessor_sieves import EntityPreprocessorSieves

import progressbar
import logging
import re
from typing import Dict, Union, List


MESSAGES = {
    "progress": "PROGRESS: Processing %(value)d-th (%(percentage)d %%) entity (in: %(elapsed)s).",
    "step": "{0}: Merging entities using {1}:"
}


class Sieve:
    """
    MergeStep is a general class for implementation of merge steps in multi-step merging approach (SIEVE_BASED).
    """

    logger = LOGGER

    def __init__(self, name: str, code: str, config: SievesParams, entity_dict: Dict[str, Entity],
                 ent_preprocessor: EntityPreprocessorSieves,
                 model=None):
        """

        Args:
            name: A name of a merge step.
            code: A code of a merge step.
            config: A config parameters required for each step.
            entity_dict: A dictionary with all entities yet to merge with each other.
            ent_preprocessor: A class that contain different properties from candidates to be merged.
            model: A word embedding model, e.g., word2vec.
        """

        self.logger.info(MESSAGES["step"].format(code, " ".join(name.split("_")[1:])))

        self.step_name = name
        self.step_code = code
        self.table = config.tables[code]

        try:
            self.params = getattr(config.params, code)
        except AttributeError:
            self.params = None

        self.steps_to_execute = config.params.steps_to_execute
        self.entity_dict = entity_dict
        self.model = model
        self.ent_preprocessor = ent_preprocessor

    def __repr__(self):
        return self.step_name

    def merge(self) -> Dict[str, Entity]:
        """
        The main execution method for each step that needs to be ovewritten. The method is supposed to reduce the number
        of entities in the dictionary of entities. The method can be either fully ovewritten or it can call
        "iterate_over_entities" method to merge entities by "winner takes it all" strategy.

        To remove smaller absorbed entities from the overall list, use "update_entity_queue" method.

        Returns:
            An updated dictionary with entities.

        """
        raise NotImplementedError

    def iterate_over_entities(self) -> Dict[str, Entity]:
        """
        The method is a "winner takes it all" strategy implementation. It iterates over the dict of entities sorted by
        descreasing entity size and finds similar entities to the one currently considered.

        Reimplement "find_similar_entities" method to specify by which strategy smaller entities will be checked for
        similarity and, if similar, merged into the bigger entity.

        Returns: Updated dict of entities.

        """
        widgets = [progressbar.FormatLabel(MESSAGES["progress"])]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(self.entity_dict) + 1, redirect_stdout=True).start()
        i = 0

        while self.entity_dict[list(self.entity_dict.keys())[0]].last_step != self.step_name \
                and i <= len(self.entity_dict):
            key = list(self.entity_dict.keys())[0]
            considered_entity = self.entity_dict[key]
            self.merge_smaller_entities(considered_entity)
            i += 1
            bar.update(i)

        self.entity_dict = {key: value for (key, value) in sorted(self.entity_dict.items(), reverse=True,
                                                                  key=lambda x: len(x[1].members))}
        bar.finish()
        return self.entity_dict

    def merge_smaller_entities(self, big_entity):
        """
        A core of each merge step: it starts a method that finds similar entities to a given one and merges the
        identified similar entities into it. find_similar_entities is to be overwritten for each implementation of
        "winner takes it all" merging strategy.

        Args:
            big_entity: A big "winner" entity, to which similar to it entities will be merged.

        """
        big_entity, to_remove_from_queue = self.find_similar_entities(big_entity)
        self.update_entity_queue(big_entity, to_remove_from_queue)

    def find_similar_entities(self, big_entity: Entity) -> (Entity, Union[Dict, List]):
        """
        A method to be overwritten for each merge step, where the step is implemented as a "winner takes it all"
        strategy.

        Args:
            big_entity: entity to be a "winner' in the "winner takes it all" strategy: if similar, the smaller entities
            will be merged into this "winner" entity.

        Returns:
            big_entity: the updated entity (smaller entities are absorbed)
            dict/list: a list with names of absorbed smalled entities;
                        if dict then key = small entity name, value = a similarity value between the big and a small
                        entity

        """
        raise NotImplementedError

    def update_entity_queue(self, updated_entity: Entity, to_remove_from_queue: Union[List, Dict],
                            step_name: str = None, update_counter: bool = True) -> str:
        """
        Updates entity dictionary (entity_dict) by removing smaller entities from it and updated parameters of the
        big entity that has absorbed the smaller ones.

        Args:
            updated_entity: A big "winner" entity to be updated.
            to_remove_from_queue: A list/dict with keys of smaller entities that need to be removed from entity_dict.
            step_name: A name of the merge step after which we update the "winner" entity.
            update_counter: A flag indicating if a name of the merge step should be recorded in the "winner" entity.

        Returns:
            An updated name of the "winner" entity.

        """
        if step_name is None:
            step_name = self.step_name

        # remove the entity from the queue
        self.entity_dict.pop(updated_entity.name)

        # smaller merged entities need to be removed from queue
        if len(to_remove_from_queue) > 0:

            for key in list(set(to_remove_from_queue)):
                try:
                    self.entity_dict.pop(key)
                except KeyError:
                    continue

            updated_entity.update_entity(step_name, update_counter, remove_oov=self.remove_oov)

        # add the same entity to the end of the queue
        self.entity_dict[updated_entity.name] = updated_entity
        return updated_entity.name

    def remove_oov(self, tokens: Union[List[Token], Set[Token]], is_head: bool) -> List[str]:
        """
        The method checks the words against a word embedding model and:
            1) removes words if they are out of vocabulary (oov) words
            2) merges them into phrases if some word collocations are present in the model as a frequent phrase,
                e.g., "illegal_aliens"
        Args:
            tokens: list of tokens
            is_head: if the list of tokens is constructed our of head words

        Returns:
            A list of words as strings that are definitely present in the word embedding model.

        """
        new_representatives = []
        representatives = []
        local_token_dict = {}

        for token in tokens:
            if token.word in LocalDictLists.stopwords or token.word in LocalDictLists.pronouns:
                continue

            if token.word in self.ent_preprocessor.phrase_ner_dict and token.ner != NON_NER:
                representatives.append(token.word)
                local_token_dict[token.word] = token
            else:
                representatives.append(token.word.lower())
                local_token_dict[token.word.lower()] = token

        for i, repr_word in enumerate(set(representatives)):
            if repr_word in self.model:
                new_representatives.append(repr_word)

            elif repr_word.lower() in self.model:
                new_representatives.append(repr_word.lower())
                local_token_dict[repr_word.lower()] = local_token_dict[repr_word]

            elif repr_word.title() in self.model:
                new_representatives.append(repr_word.title())
                local_token_dict[repr_word.title()] = local_token_dict[repr_word]

            elif not is_head:
                if repr_word.replace("-", "") in self.model:
                    new_representatives.append(repr_word.replace("-", ""))
                    local_token_dict[repr_word.replace("-", "")] = local_token_dict[repr_word]
                    continue

                for part_repr in repr_word.split("-"):
                    if part_repr in self.model:
                        new_representatives.append(part_repr)
                        local_token_dict[part_repr] = local_token_dict[repr_word]

        if len(new_representatives) > 1:
            adj = []
            nn = []
            merged_words = new_representatives.copy()
            for repr_word in new_representatives:
                if bool(re.match(JJ, local_token_dict[repr_word].pos)):
                    adj.append(repr_word)
                if bool(re.match(NN, local_token_dict[repr_word].pos)):
                    nn.append(repr_word)

            for a in adj:
                for n in nn:
                    if a + "_" + n in self.model:
                        merged_words.remove(a)
                        merged_words.remove(n)
                        merged_words.append(a + "_" + n)
                        return merged_words
            for n_r in reversed(nn):
                for n in nn:
                    if n_r + "_" + n in self.model:
                        if n_r in merged_words:
                            merged_words.remove(n_r)
                        if n in merged_words:
                            merged_words.remove(n)
                        merged_words.append(n_r + "_" + n)
                        return merged_words
            for a_r in reversed(adj):
                for a in adj:
                    if a_r + "_" + a in self.model:
                        if a_r in merged_words:
                            merged_words.remove(a_r)
                        if a in merged_words:
                            merged_words.remove(a)
                        merged_words.append(a_r + "_" + a)
                        return merged_words
            return merged_words
        else:
            return new_representatives

    @staticmethod
    def words_to_tokens(words: Union[List[str], Set[str]], token_dict: Dict[str, Token]) -> List[Token]:
        """
        The function converts a list of words into a list of tokens.
        Args:
            words: List of words.
            token_dict: Dictionary that maps words to tokens.

        Returns: List of Tokens

        """
        return [token_dict[w] for w in words if w in token_dict]
