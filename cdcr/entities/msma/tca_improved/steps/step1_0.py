from cdcr.entities.msma.step import MergeStep
from cdcr.entities.const_dict_global import *
from typing import Dict
from cdcr.entities.msma.tca_improved.entity_tca import EntityTCA

import numpy as np


class TCAStep0(MergeStep):
    """
    A step merges entities based on the matching representative head words from coreference resolution.
    """

    def __init__(self, step, docs, entity_dict: Dict[str, EntityTCA], ent_preprocessor, model):

        super().__init__(MSMA1_0, step, docs.configuration.entity_identifier_config, entity_dict, ent_preprocessor,
                         model)

        for ent in list(entity_dict.values()):
            repr = [m.head_token.word for m in ent.members if m.is_representative]
            if len(repr):
                ent.wv_attribute = repr[0]
            else:
                repr = [m.head_token.word for m in ent.members]
                ent.wv_attribute = list({k: v for k, v in ent.word_dict.items() if k in repr})[0]

    def merge(self) -> dict:
        return self.iterate_over_entities()

    def find_similar_entities(self, entity_current):
        repr_current = entity_current.wv_attribute
        type_current = entity_current.type

        to_remove_from_queue = {}

        if len(self.table[entity_current.type].values[np.where(self.table[entity_current.type] > 0)]) > 0:

            for key, entity_next in list(self.entity_dict.items())[1:]:
                if entity_next.last_step == self.step_name or len(entity_current.members) < len(entity_next.members):
                    continue

                repr_next = entity_next.wv_attribute
                type_next = entity_next.type

                if len(repr_current) == 0 or len(repr_next) == 0:
                    continue

                if self.table[type_current][type_next] == 1.0 and repr_current[0] == repr_next[0]:
                    to_remove_from_queue[key] = self.table[type_current][type_next]

        for key, sim in to_remove_from_queue.items():
            entity_current.absorb_entity(self.entity_dict[key], self.step_name, sim)

        return entity_current, to_remove_from_queue
