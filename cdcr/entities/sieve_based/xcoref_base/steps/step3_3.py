from cdcr.entities.sieve_based.step import Sieve
from cdcr.entities.const_dict_global import *

import numpy as np
import pandas as pd
import progressbar
from sklearn.cluster import AgglomerativeClustering


CLUSTERS = "clusters"
MESSAGES = {
    "vector_progress": "PROGRESS: Created word vectors for %(value)d/%(max_value)d (%(percentage)d %%) entities (in: %(elapsed)s).",
    "entities": "Forming final entities: ",
    "no_non_ne": "No suitable non-NE entities found. The step is skipped."
}


class XCoref_Base_Step3(Sieve):
    """
    A step merges person-nns, group, and person-nes entities using similarity of thier modifiers.
    """
    def __init__(self, step, docs, entity_dict, ent_preprocessor, model):

        super().__init__(XCOREF_BASE_3, step, docs.configuration.entity_identifier_config, entity_dict, ent_preprocessor,
                         model)

    def merge(self) -> dict:
        entity_types = list(self.table[self.table >= 1].stack().reset_index()["level_0"].values)
        non_ne_entities_init = {key: entity for key, entity in self.entity_dict.items()
                           if entity.type in entity_types}

        if not len(non_ne_entities_init):
            self.logger.info(MESSAGES["no_non_ne"])
            return self.entity_dict

        widgets = [
            progressbar.FormatLabel(MESSAGES["vector_progress"])
        ]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      maxval=len(non_ne_entities_init)).start()

        vector_df = pd.DataFrame(columns=["d" + str(i) for i in range(self.model.vector_size)])
        for i, (ent_key, entity) in enumerate(non_ne_entities_init.items()):
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

        if len(vector_df) < 2:
            self.logger.info(MESSAGES["no_non_ne"])
            return self.entity_dict

        vector_df.dropna(inplace=True)
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