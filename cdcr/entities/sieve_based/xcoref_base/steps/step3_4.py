from cdcr.entities.sieve_based.step import Sieve
from cdcr.entities.const_dict_global import *

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs


class XCoref_Base_Step4(Sieve):
    """
    A step merges misc abstrct entities using similarity of phrases' modifiers.
    """
    def __init__(self, step, docs, entity_dict, ent_preprocessor, model):

        super().__init__(XCOREF_BASE_4, step, docs.configuration.entity_identifier_config, entity_dict, ent_preprocessor,
                         model)

    def merge(self) -> dict:
        return self.iterate_over_entities()

    def find_similar_entities(self, entity_current):
        repr_current = list(entity_current.headwords_cand_tree)
        vector_current = self.model.query(repr_current)
        type_current = entity_current.type

        to_remove_from_queue = {}

        if len(self.table[entity_current.type].values[np.where(self.table[entity_current.type] > 0)]) > 0:

            for key, entity_next in list(self.entity_dict.items())[1:]:
                if entity_next.last_step == self.step_name or len(entity_current.members) < len(entity_next.members):
                    continue

                repr_next = list(entity_next.headwords_cand_tree)
                vector_next = self.model.query(repr_next)
                type_next = entity_next.type

                if self.table[type_current][type_next] == 0:
                    continue

                if len(repr_current) == 0 or len(repr_next) == 0:
                    continue

                sim = cs(vector_current, vector_next)[0][0]

                if type_next == type_current and self.table[type_current][type_next] == 1:
                    if sim >= self.params.min_sim_same_type:
                        to_remove_from_queue[key] = sim

                elif type_next != type_current and self.table[type_current][type_next] > 1:
                    if sim >= self.params.min_sim_diff_type:
                        to_remove_from_queue[key] = sim

        for key, sim in to_remove_from_queue.items():
            entity_current.absorb_entity(self.entity_dict[key], self.step_name, sim)

        return entity_current, to_remove_from_queue
