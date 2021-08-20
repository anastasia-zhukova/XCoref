from cdcr.entities.sieve_based.step import Sieve
from cdcr.entities.const_dict_global import *

import numpy as np


class TCAStep3(Sieve):
    """
    A step merges entities using similar compound phrases.
    """

    def __init__(self, step, docs, entity_dict, ent_preprocessor, model):

        super().__init__(TCA_3, step, docs.configuration.entity_identifier_config, entity_dict, ent_preprocessor,
                         model)

        for ent in list(self.entity_dict.values()):
            # compound_phrases = [self.remove_oov(MergeStep.words_to_tokens(list(phrase), ent.token_dict),
            compound_phrases = [self.model.optimize_phrase(Sieve.words_to_tokens(list(phrase), ent.token_dict),
                                                           True) for phrase in ent.compound_phrases]
            ent.wv_attribute = list(filter(lambda x: len(x), compound_phrases))

    def merge(self) -> dict:
        return self.iterate_over_entities()

    def find_similar_entities(self, entity_current):
        compounds_current = list(entity_current.compound_dict.keys())
        type_current = entity_current.type
        to_remove_from_queue = {}

        if len(self.table[entity_current.type].values[np.where(self.table[entity_current.type] > 0)]) > 0:
            for key_next, entity_next in list(self.entity_dict.items())[1:]:
                type_next = entity_next.type
                compounds_next = list(entity_next.compound_dict.keys())

                if entity_next.last_step == self.step_name or len(entity_current.members) < len(entity_next.members)\
                        or key_next in to_remove_from_queue:
                    continue

                if self.table[type_current][type_next] == 0:
                    continue

                all_ner = []
                for ner_type, ner_list in self.ent_preprocessor.ner_dict.items():
                    all_ner.extend(ner_list)

                matching_labeling = []
                for cur in compounds_current:
                    for nex in compounds_next:
                        if cur == nex and cur in all_ner:
                            matching_labeling.append(cur)

                matching_compound_root = TCAStep3.find_overlaps(entity_current, entity_next)
                if len(matching_compound_root) == 0:
                    matching_compound_root = TCAStep3.find_overlaps(entity_next, entity_current)

                if len(matching_compound_root) > 1:
                    to_remove_from_queue[key_next] = 1.0
                elif len(matching_compound_root) == 1:
                    for s in entity_current.representative_wordsets:
                        if frozenset(matching_compound_root).issubset(s):
                            to_remove_from_queue[key_next] = 1.0

                if len(matching_labeling) == 0:
                    continue

                # get all collocations with this compound
                compound_phrases_current = TCAStep3.phrases_with_label(entity_current.wv_attribute,
                                                                       matching_labeling)
                compound_phrases_next = TCAStep3.phrases_with_label(entity_next.wv_attribute,
                                                                    matching_labeling)

                sim_matrix = np.empty((0, len(compound_phrases_next)))
                for phrase1 in compound_phrases_current:
                    sim_array = np.empty(0)
                    for phrase2 in compound_phrases_next:
                        sim = self.model.n_similarity(phrase1, phrase2)
                        if sim >= self.table[type_current][type_next]:
                            sim_value = 1.0
                            sim_array = np.append(sim_array, sim_value)
                        else:
                            sim_array = np.append(sim_array, 0)
                    sim_matrix = np.append(sim_matrix, [sim_array], axis=0)

                sim_number = np.sum(sim_matrix[np.where(sim_matrix > 0)])
                if sim_matrix.shape[0] * sim_matrix.shape[1] == 0:
                    continue
                sim_value = sim_number / (sim_matrix.shape[0] * sim_matrix.shape[1])

                if sim_value >= self.params.merge_compound_proportion_threshold:
                    to_remove_from_queue[key_next] = {sim_value: sim_matrix}

        for key, sim in to_remove_from_queue.items():
            entity_current.absorb_entity(self.entity_dict[key], self.step_name, sim)

        return entity_current, to_remove_from_queue

    @staticmethod
    def find_overlaps(ent1, ent2):
        array = []
        # if len(ent1.adjective_labeling) == 0:
        #     return array
        for root in list(ent1.headwords_phrase_tree.keys()):
            if root in list(ent2.compound_dict.keys()):
                array.append(root)
        for adj in list(ent1.adjective_dict.keys()):
            if adj in list(ent2.compound_dict.keys()) \
                    or adj.capitalize() in list(ent2.compound_dict.keys()):
                array.append(adj)
        return array

    @staticmethod
    def phrases_with_label(compound_phrases, labeling):
        output = []
        for phrase in compound_phrases:
            for w in phrase:
                if w in labeling:
                    output.append(phrase)
        return output
