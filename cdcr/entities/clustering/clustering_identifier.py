from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as cs
import progressbar

from cdcr.entities.identifier import Identifier
from cdcr.structures.entity_set import EntitySet
from cdcr.structures.entity import Entity
from cdcr.entities.entity_preprocessor import EntityPreprocessor
from cdcr.util.magnitude_wordvectors import MagnitudeModel
from cdcr.config import *
from cdcr.entities.const_dict_global import *

CLUSTERS = "clusters"
MESSAGES = {
    "vector_progress": "PROGRESS: Created word vectors for %(value)d/%(max_value)d (%(percentage)d %%) entities (in: %(elapsed)s).",
    "entities": "Forming final entities: "
}


class ClusteringIdentifier(Identifier):

    logger = LOGGER

    def __init__(self, docs):
        super().__init__(docs)

        self.params = docs.configuration.entity_identifier_config.params
        self.model = MagnitudeModel(docs.configuration.entity_identifier_config.word_vectors)

    def extract_entities(self) -> EntitySet:
        ent_preprocessor = EntityPreprocessor(self.docs, Entity)
        self.entity_dict = ent_preprocessor.entity_dict_construction()

        widgets = [
            progressbar.FormatLabel(MESSAGES["vector_progress"])
        ]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      maxval=len(self.entity_dict)).start()

        vector_df = pd.DataFrame(columns=["d" + str(i) for i in range(self.model.vector_size)])
        for i, (ent_key, entity) in enumerate(self.entity_dict.items()):
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
        #
        # cluster_alg = AgglomerativeClustering(n_clusters=None,
        #                                       affinity="cosine", compute_full_tree=True,
        #                                       distance_threshold=self.params.min_sim, linkage="average")
        cluster_alg = AffinityPropagation(damping=0.7)
        clusters = cluster_alg.fit(vector_df)
        cluster_df = pd.DataFrame(clusters.labels_, columns=[CLUSTERS], index=vector_df.index)
        cluster_df.sort_values(by=[CLUSTERS], inplace=True)
        entity_dict_updated = {}
        self.logger.info(MESSAGES["entities"])

        for cluster_id in sorted(list(set(cluster_df[CLUSTERS].values))):
            df_local = cluster_df[cluster_df[CLUSTERS] == cluster_id]
            entity_keys = list(df_local.index)

            if not len(entity_keys):
                continue

            if len(entity_keys) == 1:
                key = entity_keys[0]
                entity_dict_updated[key] = self.entity_dict[key]
                self.logger.info("Entity: " + key)
            else:
                main_ent = self.entity_dict[entity_keys[0]]

                for ent_key in entity_keys[1:]:
                    self.logger.info(main_ent.name + "  <--  " + self.entity_dict[ent_key].name)
                    main_ent.add_members(self.entity_dict[ent_key].members)

                main_ent.update_entity(CLUSTERING)
                entity_dict_updated[main_ent.name] = main_ent
                self.logger.info("Entity: " + main_ent.name)

        entity_set = EntitySet(identification_method=self.docs.configuration.entity_method, topic=self.docs.topic)
        entity_set.extend(list(entity_dict_updated.values()))
        entity_set.sort(reverse=True, key=lambda x: len(x.members))
        return entity_set

