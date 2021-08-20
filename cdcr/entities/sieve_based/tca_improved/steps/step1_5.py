from cdcr.entities.sieve_based.step import Sieve
from cdcr.entities.const_dict_global import *

import numpy as np
import Levenshtein


class TCAStep5(Sieve):
    """
    A step merges entities using similar representative phrases.
    """

    def __init__(self, step, docs, entity_dict, ent_preprocessor, model):

        super().__init__(TCA_5, step, docs.configuration.entity_identifier_config, entity_dict, ent_preprocessor,
                         model)

    def merge(self) -> dict:
        return self.iterate_over_entities()

    def find_similar_entities(self, entity_current):
        to_remove_from_queue = {}
        to_remove_from_queue = self.find_matching_strings(entity_current, to_remove_from_queue, 1)

        for key, sim in to_remove_from_queue.items():
            entity_current.absorb_entity(self.entity_dict[key], self.step_name, sim)

        return entity_current, to_remove_from_queue

    def find_matching_strings(self, entity_current, to_remove_from_queue, recursion_level):
        core_string_current = entity_current.representative_phrases["all"]
        type_current = entity_current.type

        for key_next, entity_next in list(self.entity_dict.items())[1:]:
            if entity_next.last_step == self.step_name or len(entity_current.members) < len(entity_next.members)\
                    or key_next in to_remove_from_queue:
                continue

            if self.table[entity_current.type][entity_next.type] == 0:
                continue

            core_string_next = entity_next.representative_phrases["all"]
            type_next = entity_next.type

            if len(core_string_current) == 0 or len(core_string_next) == 0:
                continue

            sim_matrix = np.empty((0, len(core_string_next)))

            for phrase_current in core_string_current:
                sim_array = np.empty(0)
                phrase_current_str = " ".join([t.word for t in phrase_current])
                for phrase_next in core_string_next:
                    max_str = np.max([len(phrase_current), len(phrase_next)])
                    phrase_next_str = " ".join([t.word for t in phrase_next])
                    sim = Levenshtein.distance(phrase_current_str, phrase_next_str)

                    if phrase_current_str.replace(".", "") == phrase_next_str.replace(".", ""):
                        sim = 0

                    sim_value = sim / max_str
                    if sim_value <= self.table[type_current][type_next]:
                        sim_value = 2.0 if sim <= self.table[type_current][type_next] - self.params.very_similar_add_value \
                            else 1.0

                        sim_array = np.append(sim_array, sim_value)

                    else:
                        sim_array = np.append(sim_array, 0)
                sim_matrix = np.append(sim_matrix, [sim_array], axis=0)

            sim_num_hor = np.max(np.sum(sim_matrix > 0, axis=1))
            size_hor = sim_matrix.shape[1]
            sim_num_vert = np.max(np.sum(sim_matrix > 0, axis=0))
            size_vert = sim_matrix.shape[0]
            sim_value = 0
            if sim_num_hor >= sim_num_vert:
                if size_hor > 1:
                    sim_value = sim_num_hor / size_hor
            else:
                if size_vert > 1:
                    sim_value = sim_num_vert / size_vert
            if sim_value >= self.params.merge_string_proportion_threshold:
                to_remove_from_queue[key_next] = sim_value

                if recursion_level == 0:
                    return to_remove_from_queue
                else:
                    to_remove_from_queue = self.find_matching_strings(entity_next, to_remove_from_queue, 0)

        return to_remove_from_queue
