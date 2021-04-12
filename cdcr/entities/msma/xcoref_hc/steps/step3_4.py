from cdcr.entities.msma.step import MergeStep
from cdcr.entities.const_dict_global import *

import numpy as np
import pandas as pd
import progressbar
from sklearn.cluster import AgglomerativeClustering


CLUSTERS = "clusters"
MESSAGES = {
    "vector_progress": "PROGRESS: Created word vectors for %(value)d/%(max_value)d (%(percentage)d %%) entities (in: %(elapsed)s).",
    "entities": "Forming final entities: "
}


class XCoref_HC_Step4(MergeStep):
    """
    A step merges misc abstrct entities using similarity of phrases' modifiers.
    """
    def __init__(self, step, docs, entity_dict, ent_preprocessor, model):

        super().__init__(MSMA3_4, step, docs.configuration.entity_identifier_config, entity_dict, ent_preprocessor,
                         model)

    def merge(self) -> dict:
        entity_types = list(self.table[self.table >= 1].stack().reset_index()["level_0"].values)
        misc_entities_init = {key: entity for key, entity in self.entity_dict.items()
                           if entity.type in entity_types}

        widgets = [
            progressbar.FormatLabel(MESSAGES["vector_progress"])
        ]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      maxval=len(misc_entities_init)).start()

        vector_df = pd.DataFrame(columns=["d" + str(i) for i in range(self.model.vector_size)])
        for i, (ent_key, entity) in enumerate(misc_entities_init.items()):
            entity_array = np.zeros((0, self.model.vector_size))
            for m in entity.members:

                if m.coref_subtype == PRONOMINAL:
                    continue

                vector = self.model.query(m.tokens)
                entity_array = np.append(entity_array, vector, axis=0)
            vector_df = vector_df.append(pd.DataFrame([np.mean(entity_array, axis=0)],
                                                      columns=["d" + str(i) for i in range(self.model.vector_size)],
                                                      index=[ent_key]))
            bar.update(i+1)
        bar.finish()

        cluster_alg = AgglomerativeClustering(n_clusters=None,
                                              affinity="cosine", compute_full_tree=True,
                                              distance_threshold=self.params.min_sim, linkage="average")
        clusters = cluster_alg.fit(vector_df)
        cluster_df = pd.DataFrame(clusters.labels_, columns=[CLUSTERS], index=vector_df.index)
        cluster_df.sort_values(by=[CLUSTERS], inplace=True)
        self.logger.info(MESSAGES["entities"])

        for cluster_id in sorted(list(set(cluster_df[CLUSTERS].values))):
            df_local = cluster_df[cluster_df[CLUSTERS] == cluster_id]
            entity_keys = list(df_local.index)

            if not len(entity_keys):
                continue

            if len(entity_keys) == 1:
                key = entity_keys[0]
                self.update_entity_queue(self.entity_dict[key], [], self.step_name, True)
            else:
                main_ent = self.entity_dict[entity_keys[0]]

                for key in entity_keys[1:]:
                    main_ent.absorb_entity(self.entity_dict[key], self.step_name, 1.0)

                self.update_entity_queue(main_ent, entity_keys[1:], self.step_name, True)

        return {key: value for (key, value) in sorted(self.entity_dict.items(), reverse=True,
                                                                              key=lambda x: len(x[1].members))}