from cdcr.entities.msma.step import MergeStep
from cdcr.entities.const_dict_global import *

import numpy as np

SIM_ELEMENTS = "sim_elements"
SIM = "sim"


class TCAStep2(MergeStep):
    """
    A step merges entities using similar representative labeling phrases.
    """

    def __init__(self, step, docs, entity_dict, ent_preprocessor, model):

        super().__init__(MSMA1_2, step, docs.configuration.entity_identifier_config, entity_dict, ent_preprocessor,
                         model)

        for ent in list(self.entity_dict.values()):
            wv_labeling = list(filter(lambda x: len(x),
                                      # [self.remove_oov(MergeStep.words_to_tokens(list(phrase), ent.token_dict), False)
                                      [self.model.optimize_phrase(MergeStep.words_to_tokens(list(phrase), ent.token_dict), True)
                                       for phrase in list(ent.adjective_phrases)]))
            ent.wv_attribute = list(filter(lambda x: len(x), wv_labeling))
            ent.representative_labeling_calc(ent_preprocessor, self.model)

        self.merged = []

    def merge(self) -> dict:
        return self.iterate_over_entities()

    def find_similar_entities(self, entity_current):
        to_remove_from_queue = []
        # check if there any possible comparisons available for this entity type
        if len(self.table[entity_current.type].values[np.where(self.table[entity_current.type] > 0)]) > 0:
            to_remove_from_queue = self.sim_labeling_recursion(entity_current)
            entity_current = self.absorb_entities(entity_current, to_remove_from_queue)

        return entity_current, to_remove_from_queue

    def sim_labeling_recursion(self, entity_current):
        labeling_current = entity_current.representative_labeling
        type_current = entity_current.type
        key_current = entity_current.name
        sim_dict = {}

        for key_next, entity_next in list(self.entity_dict.items())[1:]:
            type_next = entity_next.type
            labeling_next = entity_next.representative_labeling

            if entity_next.last_step == self.step_name  or len(entity_current.members) < len(entity_next.members) \
                    or key_next in self.merged or key_current == key_next:
                continue

            if self.table[type_current][type_next] == 0:
                continue

            sim_matrix = np.empty((0, len(labeling_next)))
            for phrase1 in labeling_current:
                sim_array = np.empty(0)
                for phrase2 in labeling_next:
                    sim = self.model.n_similarity(phrase1, phrase2)
                    if sim >= self.table[type_current][type_next]:
                        sim_value = 2.0 if sim >= self.table[type_current][type_next] + \
                                           self.params.very_similar_add_value else 1.0
                        sim_array = np.append(sim_array, sim_value)
                    else:
                        sim_array = np.append(sim_array, 0)
                sim_matrix = np.append(sim_matrix, [sim_array], axis=0)
            sim_number = np.sum(sim_matrix[np.where(sim_matrix > 0)])

            if sim_matrix.shape[0] * sim_matrix.shape[1] == 0:
                continue

            dim_number = sim_matrix.shape[0] * sim_matrix.shape[1]
            sim_value = sim_number / dim_number
            if dim_number <= self.params.merge_labeling_min_dim_number and type_next != type_current:
                continue

            if sim_value >= self.params.merge_labeling_proportion_threshold:
                sim_dict[key_next] = {SIM: {sim_value: sim_matrix}, SIM_ELEMENTS: {}}
                self.merged.append(key_next)

        if len(sim_dict) > 0:
            # recursion
            for sim_ent in list(sim_dict.keys()):
                sim_dict[sim_ent][SIM_ELEMENTS] = self.sim_labeling_recursion(self.entity_dict[sim_ent])
            return sim_dict
        else:
            return {}

    def absorb_entities(self, entity, queue):
        for key, sim_params in queue.items():
            if len(sim_params[SIM_ELEMENTS]) > 0:
                inner_entity = self.absorb_entities(self.entity_dict[key], sim_params[SIM_ELEMENTS])
                entity.absorb_entity(inner_entity, self.step_name, list(sim_params[SIM])[0])
            else:
                entity.absorb_entity(self.entity_dict[key], self.step_name, list(sim_params[SIM])[0])
        return entity
