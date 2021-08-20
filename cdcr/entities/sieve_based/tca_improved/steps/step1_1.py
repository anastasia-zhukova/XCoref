from cdcr.entities.sieve_based.step import Sieve
from cdcr.entities.const_dict_global import *

import pandas as pd
import numpy as np
import math


class TCAStep1(Sieve):

    def __init__(self, step, docs, entity_dict, ent_preprocessor, model):
        """
        A step merges entities using similar headwords in the vector space.
        """

        super().__init__(TCA_1, step, docs.configuration.entity_identifier_config, entity_dict, ent_preprocessor,
                         model)

        for ent in list(self.entity_dict.values()):
            # cleaned_heads = self.remove_oov(MergeStep.words_to_tokens(list(ent.headwords_cand_tree), ent.token_dict),
            cleaned_heads = self.model.optimize_phrase(Sieve.words_to_tokens(list(ent.headwords_cand_tree), ent.token_dict),
                                                       True)
            if len(set([k.lower() for k in cleaned_heads])) == len(cleaned_heads):
                ent.wv_attribute = cleaned_heads
            else:
                PART, COUNT = "part", "count"
                df = pd.DataFrame(columns=[PART, COUNT])
                for head in cleaned_heads:
                    try:
                        count = len(ent.headwords_cand_tree[head])
                    except KeyError:
                        count = math.floor(np.mean(list(ent.word_dict.values())))

                    df = df.append(pd.DataFrame({
                        PART: head[1:],
                        COUNT: count
                    }, index=[head]))

                df_sorted = df.sort_values(by=[PART, COUNT], ascending=[True, False])
                df_final = df_sorted.drop_duplicates(subset=[PART], keep="first")
                ent.wv_attribute = list(df_final.index)

    def merge(self) -> dict:
        return self.iterate_over_entities()

    def find_similar_entities(self, entity_current):
        core_current = entity_current.wv_attribute
        type_current = entity_current.type
        to_remove_from_queue = {}

        for key, entity_next in list(self.entity_dict.items())[1:]:
            if entity_next.last_step == self.step_name or len(entity_current.members) < len(entity_next.members):
                continue

            core_next = entity_next.wv_attribute
            type_next = entity_next.type

            if len(core_current) == 0 or len(core_next) == 0:
                continue

            if self.table[type_current][type_next] == 0:
                continue

            sim = self.model.n_similarity(core_current, core_next)

            if sim >= self.table[type_current][type_next]:
                to_remove_from_queue[key] = sim

        for key, sim in to_remove_from_queue.items():
            entity_current.absorb_entity(self.entity_dict[key], self.step_name, sim)

        return entity_current, to_remove_from_queue
