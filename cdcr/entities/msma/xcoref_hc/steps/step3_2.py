from cdcr.entities.msma.step import MergeStep
from cdcr.entities.const_dict_global import *
from cdcr.entities.dict_lists import LocalDictLists

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as cs
import string


MESSAGES = {
    "vectors": "Comparison table row {0}/{1}: Building vectors for {2} entities. ",
    "vectors2": "                        : Building vectors for {} entities. ",
    "sim": "Calculating similarities"
}


class MSMA3Step2(MergeStep):
    """
    A step merges entities using similarity of core mentions to non-NE mentions in cores for ELMO.
    """

    def __init__(self, step, docs, entity_dict, ent_preprocessor, model):

        super().__init__(MSMA3_2, step, docs.configuration.entity_identifier_config, entity_dict, ent_preprocessor,
                         model)

        self.vector_df = pd.DataFrame(columns=["d" + str(i) for i in range(self.model.vector_size)])

    def merge(self) -> dict:
        for i, (index_current, row) in enumerate(self.table.iterrows()):
            if "-ne" not in index_current:
                continue

            self.logger.info(MESSAGES["vectors"].format(str(i), str(len(self.table)), index_current))

            entities_big = {k:v for k,v in self.entity_dict.items() if v.type == index_current}
            vector_df_big = self._build_vectors(entities_big)

            entities_small = {}
            smaller_list = []
            for index_next in list(self.table.columns):
                if row[index_next] == 0:
                    continue
                smaller_list.append(index_next)
                entities_small.update({k: v for k, v in self.entity_dict.items() if v.type == index_next
                                       and all([m.head_token.ner != PERSON_NER for m in v.members])})

            self.logger.info(MESSAGES["vectors2"].format(", ".join(smaller_list)))
            vector_df_small = self._build_vectors(entities_small)

            self.logger.info(MESSAGES["sim"])
            sim_df = pd.DataFrame(np.zeros((len(vector_df_big), len(vector_df_small))), columns=list(vector_df_small.index),
                                  index=list(vector_df_big.index))

            for index_big, row_big in vector_df_big.iterrows():
                for index_small, row_small in vector_df_small.iterrows():
                    if index_big == index_small:
                        continue

                    entity_current = entities_big[index_big]
                    repr_current = list(set([m.text.replace("_", " ") for m in entity_current.members]))
                    repr_current_heads = set([v.lower() for v in list(entity_current.headwords_cand_tree)]).union(set([
                        v.lower() for v in list(entity_current.appos_dict)]))

                    entity_next = entities_small[index_small]
                    repr_next = list(set([m.text.replace("_", " ") for m in entity_next.members]))
                    repr_next_heads = set([m.head_token.word.lower() for m in entity_next.members])
                    to_break = False
                    for phrase_next in repr_next:
                        for phrase_current in repr_current:
                            issubset = set(phrase_next.lower().split(" ")).issubset(
                                set(phrase_current.lower().split(" ")))
                            inters = [v for v in list(set(phrase_next.lower().split(" ")).intersection(
                                set(phrase_current.lower().split(" ")))) if v not in LocalDictLists.stopwords]
                            if issubset and (len(set(phrase_next.lower().split(" ")).intersection(set([v.lower()
                                                 for v in list(entity_current.compound_dict)])))
                                             or len(set(inters).intersection(set([v.lower()
                                                 for v in list(entity_current.appos_dict)])))
                                             or len(set(inters).intersection(set([v.lower()
                                                 for v in list(entity_next.appos_dict)])))):
                                             # or len(set([v.lower() for v in inters]).intersection(set([v.lower()
                                             #        for v in list(entity_next.adjective_dict)])))
                                             # or len(set([v.lower() for v in inters]).intersection(set([v.lower()
                                             #        for v in list(entity_current.adjective_dict)])))):
                                sim_df.loc[index_big, index_small] = 1.0
                                to_break = True
                                break
                            if ((len(inters) >= 1 and len([v for v in phrase_next.lower().split(" ")
                                          if v not in LocalDictLists.stopwords]) == 1) or len(inters) >= 2) and \
                                    len(set(inters).intersection(repr_next_heads)) >= 1 \
                                    and len(set(inters).intersection(repr_current_heads)) >= 1:
                                sim_df.loc[index_big, index_small] = 1.0
                                to_break = True
                                break
                        if to_break:
                            break
                    if not to_break:
                        sim = cs(row_big.values.reshape(1, -1), row_small.values.reshape(1, -1))[0][0]

                        if sim >= self.params.mention_similarity_threshold \
                                and self.table.loc[
                            self.entity_dict[index_big].type, self.entity_dict[index_small].type] == 1:
                            sim_df.loc[index_big, index_small] = sim

            winner_dict = {}
            for col in list(sim_df.columns):
                if np.sum(sim_df[col].values) == 0:
                    continue
                best_phrase = sim_df[col].idxmax(axis=0)
                # if best_phrase not in winner_dict:
                winner_dict[best_phrase] = winner_dict.get(best_phrase, []) + [{"key":col, "sim": sim_df[col].max(axis=0)}]

            a = 1
            for main_ent_key, values in winner_dict.items():
                main_entity = self.entity_dict[main_ent_key]
                for val in values:
                    main_entity.absorb_entity(self.entity_dict[val["key"]], self.step_name, val["sim"])
                self.update_entity_queue(main_entity, [v["key"] for v in values], self.step_name, True)

        return {key: value for (key, value) in sorted(self.entity_dict.items(), reverse=True,
                                                                              key=lambda x: len(x[1].members))}

        # return self.iterate_over_entities()
    def _build_vectors(self, entity_dict):
        columns = ["d" + str(i) for i in range(self.model.vector_size)]
        vector_df = pd.DataFrame(columns=columns)
        for key, ent in entity_dict.items():
            local_vector_df = pd.DataFrame(columns=columns)
            # phrase_set = set([" ".join([t.word for t in m.tokens if t.word not in LocalDictLists.stopwords or
            #                             t.word not in string.punctuation]) for m in ent.members])
            phrase_dict = {m.text: m.tokens for m in ent.members}
            # for phrase in ent.core_mentions:
            for phrase, phrase_tokens in phrase_dict.items():
                local_vector_df = local_vector_df.append(pd.DataFrame(self.model.query(phrase_tokens), columns=columns,
                                                          index=[phrase]))
            vector_df = vector_df.append(
                pd.DataFrame([np.mean(local_vector_df.values, axis=0)], columns=columns, index=[key]))
        return vector_df