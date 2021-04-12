from cdcr.entities.entity_preprocessor import EntityPreprocessor
from cdcr.entities.const_dict_global import *
from cdcr.entities.dict_lists import LocalDictLists
from cdcr.structures.entity_set import EntitySet
from cdcr.structures.entity import Entity

import pandas as pd
import progressbar

CAND_GROUP = "cand_group"
CLUSTER = "cluster"
ENTITIES = "entities"
NOTIFICATION_MESSAGES = {
    "progress": "PROGRESS: Build entities from %(value)d (%(percentage)d %%) coreference groups"
                " (in: %(elapsed)s)."
}


class EntityPreprocessorEECDCR(EntityPreprocessor):

    def __init__(self, docs, entity_class):

        super().__init__(docs, entity_class)

    def entity_dict_construction(self, resolved_mentions, mention_dict):
        entity_dict = {}
        df = pd.DataFrame({CLUSTER: [cl.cluster_id for cl in resolved_mentions for m in list(cl.mentions) ]},
                           index=[m for cl in resolved_mentions for m in list(cl.mentions)])
        df[CAND_GROUP] = [""] * len(df)
        for mention, cand in mention_dict.items():
            df.loc[mention, CAND_GROUP] = cand.candidate_group_set.group_name
        df[ENTITIES] = [None] * len(df)

        widgets = [
            progressbar.FormatLabel(NOTIFICATION_MESSAGES["progress"])
        ]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      maxval=len(set(df[CAND_GROUP].values))).start()

        for cand_group_id, cand_group_name in enumerate(set(df[CAND_GROUP].values)):
            df_local = df[df[CAND_GROUP] == cand_group_name]
            for cl_id in set(df_local[CLUSTER].values):
                entity_df = df_local[df_local[CLUSTER] == cl_id]
                members = []
                m_ids = []
                for m_id in list(entity_df.index):
                    cand = mention_dict[m_id]

                    if cand.annot_text == "" or cand.annot_text is None:
                        # ignore pronominals from coref groups
                        # if cand.coref_subtype == PRONOMINAL or cand.head_token.pos in ["PRP", "PRP$"]:
                        #     continue

                        # ignore NPs with only one article-word
                        if len(cand.tokens) == 1 and cand.tokens[0].pos == DT:
                            continue

                        # ignore one-word NPs consisting of single adjectives
                        if (JJ in cand.head_token.pos and cand.head_token.ner == NON_NER and
                            len([t.word for t in cand.tokens if EntityPreprocessorEECDCR.not_stopword(t.word)]) == 1) \
                                and self.config_params.preprocessing.exclude_single_adj:
                            continue

                        # ignore NPs where a head word is a general phrase, e.g., everything, nothing, etc
                        if cand.head_token.word in LocalDictLists.general_nouns \
                                and self.config_params.preprocessing.exclude_general_nouns:
                            continue

                        # ignore single-word NPs with titles Mr, Ms, etc.
                        if cand.head_token.word in LocalDictLists.titles:
                            continue

                        # ignore NPs where a root word is a words like "those", "some", etc.
                        # if cand.head_token.pos in [DT, POS] and self.config_params.preprocessing.exclude_dt:
                        #     continue

                        # ignore candidates related to time, date, and duration
                        if cand.head_token.ner in [TIME_NER, DATE_NER, DURATION_NER] and \
                                self.config_params.preprocessing.exclude_time:
                            continue

                    members.append(cand)
                    m_ids.append(m_id)

                if not len(members):
                    continue

                ent = Entity(self.docs, members, None, self, None)
                entity_dict[ent.name] = ent
                for m_id in m_ids:
                    df.loc[m_id, ENTITIES] = ent.name

            bar.update(cand_group_id + 1)
        bar.finish()

        return {k:v for k,v in sorted(entity_dict.items(), reverse=True, key=lambda x: len(x[1].members))}, df
