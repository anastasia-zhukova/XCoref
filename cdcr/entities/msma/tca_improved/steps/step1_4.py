from cdcr.entities.msma.step import MergeStep
from cdcr.entities.const_dict_global import *

import numpy as np


class TCAStep4(MergeStep):
    """
    A step merges entities using similar representative wordsets.
    """

    def __init__(self, step, docs, entity_dict, ent_preprocessor, model):

        super().__init__(MSMA1_4, step, docs.configuration.entity_identifier_config, entity_dict, ent_preprocessor,
                         model)

        for ent in list(self.entity_dict.values()):
            # freq_phrases = [self.remove_oov(MergeStep.words_to_tokens(list(phrase), ent.token_dict),
            freq_phrases = [self.model.optimize_phrase(MergeStep.words_to_tokens(list(phrase), ent.token_dict),
                                               True) for phrase in ent.representative_wordsets]
            ent.wv_attribute = list(filter(lambda x: len(x), freq_phrases))

    def merge(self) -> dict:
        return self.iterate_over_entities()

    def find_similar_entities(self, entity_current):
        to_remove_from_queue = {}
        to_remove_from_queue = self.find_freq_sim(entity_current, to_remove_from_queue, 1)

        for key, sim in to_remove_from_queue.items():
            entity_current.absorb_entity(self.entity_dict[key], self.step_name, list(sim)[0])

        return entity_current, to_remove_from_queue

    def find_freq_sim(self, entity_current, to_remove_from_queue, recursion_level):
            for key_next, entity_next in list(self.entity_dict.items())[1:]:
                if entity_next.last_step == self.step_name or len(entity_current.members) < len(entity_next.members)\
                        or key_next in to_remove_from_queue:
                    continue

                if len(entity_next.wv_attribute) == 0 \
                        or len(entity_current.wv_attribute) == 0:
                    continue

                if self.table[entity_current.type][entity_next.type] == 0:
                    continue

                sim_final = 0
                sim_matrix = np.empty((0, len(entity_next.wv_attribute)))

                for root_current in entity_current.wv_attribute:
                    sim_array = np.empty(0)

                    for root_next in entity_next.wv_attribute:
                        sim = self.model.n_similarity(root_current, root_next)
                        if sim >= self.table[entity_current.type][entity_next.type]:
                            sim_final = max(sim, sim_final)
                            sim_value = 2.0 if sim >= self.table[entity_current.type][entity_next.type] + \
                                               self.params.very_similar_add_value else 1.0
                            sim_array = np.append(sim_array, sim_value)

                        else:
                            sim_array = np.append(sim_array, 0)

                    sim_matrix = np.append(sim_matrix, [sim_array], axis=0)

                sim_number = len(sim_matrix[np.where(sim_matrix > 0)])
                merge = False
                sim_value = sim_number / (sim_matrix.shape[0] * sim_matrix.shape[1])

                if sim_value >= self.params.merge_freq_phrases_proportion:
                    merge = True

                if merge:
                    to_remove_from_queue[key_next] = {sim_value: sim_matrix}

                    if recursion_level == 0:
                        return to_remove_from_queue
                    else:
                        to_remove_from_queue = self.find_freq_sim(entity_next, to_remove_from_queue, 0)

            return to_remove_from_queue
