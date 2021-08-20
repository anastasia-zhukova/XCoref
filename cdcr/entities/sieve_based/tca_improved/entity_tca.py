from cdcr.entities.sieve_based.entity_sieves import EntitySieves

import numpy as np
import pandas as pd
from pymining import itemmining
import math
from sklearn.cluster import AffinityPropagation

NOTIFICATION_MESSAGES = {
    "no_repr":"No representative words were counted before. Run representative calculation after entity "
              "initialization.",
    "no_word_dict": "Word dictionary is empty. An entity can't be initialized."
}


class EntityTCA(EntitySieves):
    """
    An Entity class for TCA.
    """

    def __init__(self, document_set, ent_preprocessor, members, name, wikipage=None, core_mentions=None):

        self.config_params = document_set.configuration.entity_identifier_config.params.entity_property
        self.representative_word_num = self.config_params.representative_word_num

        super().__init__(document_set, ent_preprocessor, members, name, wikipage, core_mentions)

        self._labeling_extraction()
        self.representative_labeling = []
        self.representative_wordsets = []
        self.representative_wordset_calc()

        self.wv_attribute = []

    def additional_param_update(self, **kwargs):
        self._labeling_extraction()
        self.representative_wordset_calc()

    def representative_wordset_calc(self):
        """
        Calculates representative wordsets.

        """
        self.word_dict = {key: value for (key, value) in sorted(self.word_dict.items(),
                                                                reverse=True,
                                                                key=lambda x: x[1])}
        phrases_list = []
        for head, neighbors in list(self.headwords_phrase_tree.items()):
            for neighbor_set, cands in neighbors.items():
                phrase = set(neighbor_set)
                phrase.add(head)
                phrases_list.extend([tuple(phrase)] * len(cands))
        if len(phrases_list) == 1:
            # self.representative_core_set = [frozenset(phrases_list[0])]
            self.representative_wordsets = phrases_list
        else:
            relim_input = itemmining.get_relim_input(phrases_list)
            freq_sets = itemmining.relim(relim_input, min_support=min(self.config_params.min_support, len(phrases_list)))

            max_itemsets = self._max_set(freq_sets)

            if list(self.word_dict.values())[0] / len(self.members) >= self.config_params.very_freq_item_threshold:
                max_itemsets[frozenset({list(self.word_dict.keys())[0]})] = list(self.word_dict.values())[0]
            max_itemsets = {key: value for (key, value) in sorted(max_itemsets.items(), reverse=True,
                                                          key=lambda x: math.log(1 + len(x[0])) * math.log(x[1], 2))}
            self.representative_wordsets = list(max_itemsets.keys())[:min(len(max_itemsets),
                                                                          self.representative_word_num)]
        # if len(self.representative_core) == 0:
        #     self.representative_core = [list(self.labeling.keys())[0]]

    def _max_set(self, data):
        """
        Calculated maxinal frequent itemsets.

        """

        def __itemset_comparison(old_set, new_set):
            if old_set.issubset(new_set):
                return True, old_set, new_set
            if new_set.issubset(old_set):
                return False, None, None
            else:
                return False, None, new_set

        max_itemset_list = list()
        dict_string_to_set = dict()
        for itemset, number in data.items():
            split_itemset = set(itemset)
            dict_string_to_set[id(split_itemset)] = (itemset, number)
            if len(max_itemset_list) == 0:
                max_itemset_list.append(split_itemset)
                continue
            del_old, del_set, new_set = None, None, None
            for max_itemset in list(reversed(max_itemset_list)):
                del_old, del_set, new_set = __itemset_comparison(max_itemset, split_itemset)
                if del_old:
                    # found a bigger itemset, replace a small one
                    max_itemset_list.remove(del_set)
                    max_itemset_list.append(new_set)
                    break
                elif new_set is None:
                    # a bigger set already exists
                    break
            if not del_old:
                # a new itemset found
                if new_set is not None:
                    max_itemset_list.append(new_set)

        dict_output = dict()
        for maxset in max_itemset_list:
            itemset_str, number = dict_string_to_set[id(maxset)]
            dict_output[frozenset(maxset)] = number
        return dict_output

    def representative_labeling_calc(self, ent_preprocessor, model):
        """
        Calculates representative labeling phrases.

        """
        global_labeling = ent_preprocessor.labeling_dict

        def _avg_dim(array):
            return np.sum(array, axis=0) / array.shape[0]

        if len(self.wv_attribute) == 0:
            return []

        one_word_phrase = []
        CLUSTERS, COUNT, LABELING_COUNT = "clusters", "count", "labeling_count"

        # find groups of similar labeling
        full_array = np.empty((0, model.vector_size))
        for phrase in self.wv_attribute:
            local_array = np.empty((0, model.vector_size))
            if len(phrase) == 1:
                one_word_phrase.append(phrase[0])
            for word in phrase:
                local_array = np.append(local_array, [model.get_vector(word)], axis=0)
            full_array = np.append(full_array, [_avg_dim(local_array)], axis=0)

        df = pd.DataFrame(data=full_array, index=["_".join(phrase) for phrase in self.wv_attribute],
                          columns=["d" + str(i) for i in range(model.vector_size)])
        aff = AffinityPropagation(damping=self.config_params.damping, max_iter=self.config_params.max_iter)
        labels_pred = aff.fit_predict(df)
        df[CLUSTERS] = labels_pred
        df[COUNT] = [len(cands) for labeling, cands in self.adjective_phrases.items()]
        df[LABELING_COUNT] = [0] * len(df)

        for phrase in list(df.index):
            for word in phrase.split("_"):
                if word in self.adjective_dict:
                    if word in global_labeling:
                        df.loc[phrase, LABELING_COUNT] = global_labeling[word]

        # within each cluster, select as cluster representatives labeling that occurred most frequently in this entity
        cluster_representatives = []
        for cl in list(set(labels_pred)):
            entity_data_cl = df[df[CLUSTERS] == cl]
            column = LABELING_COUNT
            max_val = np.max(entity_data_cl[column])
            if max_val == 0:
                column = COUNT
                max_val = np.max(entity_data_cl[column])
            # if two labelings are equal to max value, choose the one that is used more often to label
            cluster_representatives.append(entity_data_cl[entity_data_cl[column] == max_val].iloc[0, :].name)

        self.representative_labeling = [cl.split("_") if cl not in one_word_phrase else [cl]
                                        for cl in cluster_representatives ]
