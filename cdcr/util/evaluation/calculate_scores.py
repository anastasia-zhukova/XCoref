from typing import List
import pandas as pd
import re
import os
import numpy as np
import warnings
import shortuuid
import progressbar
import string
import math
from munkres import Munkres, make_cost_matrix
from sklearn.metrics import homogeneity_completeness_v_measure as hcv
# import logging
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

from cdcr.structures.document_set import DocumentSet
from cdcr.config import LOGGER
import cdcr.logger as logging
from cdcr.config import ConfigLoader
from cdcr.candidates.cand_enums import *
from cdcr.entities.dict_lists import LocalDictLists
from candidates.cand_enums import OriginType


ENTITIES = "entities"
ORIGIN_TYPE = "origin_type"
COREF_STRATEGY = "coref_strategy"
ENTITY_METHOD = "identification_method"
PARAMS = "parameters"
WORDVECTORS = "wordvectors"
DATASET = "dataset"
TOPIC_NAME = "topic_name"
ANNOT_DATASET = "annot_dataset"
CONFIG = "config"

LABEL, CONCEPT_TYPE, PHRASE, WORDSET, HEAD = "label", "concept_type", "phrase", "wordset", "head"
WCL, NUMBER, NUMBER_ENT = "wcl", "mentions_num", "entities_num"
TIME = "time"
AVG_TIME = "avg_time"


MUC = "_MUC"
B3 = "_B3"
CEAF_M = "_CEAF_M"
CEAF_E = "_CEAF_E"
BLANC = "_BLANC"
AVG = "_AVG"
WEIGHTED = "_WEIGHTED"
WCL_WEIGHTED = "_WCL_WEIGHTED"
CONLL = "_CONLL"
SUPPORT = "Support"
H = "Hom"
C = "Compl"
V = "V-measure"
P = "P"
R = "R"
F1 = "F1"
TRUE, PRED = "_true", "_pred"
UNK = "UNK"
TRUE_LABEL, PRED_LABEL, TRUE_PHRASE, PRED_PHRASE, ANNOT_TYPE, PRED_TYPE = "label_true", "label_pred", "phrase_true", \
                                                                          "phrase_pred", "type_true", "type_pred"


class EvalScorer:

    def __init__(self, document_set_list: List[DocumentSet], modules: List[str]):
        self.document_set_list = document_set_list
        self.modules = modules
        self.datasets = {}
        self.dataset_stats = {}

    def run_evaluation(self):
        result_dfs = []
        for module in self.modules:
            if module == ENTITIES:
                result_dfs.append(self.evaluate_entities())
        return result_dfs

    def evaluate_entities(self):
        conf = ConfigLoader.load_and_apply()
        logging.setup()
        conf.log()

        # self.document_set_list[4].candidates.origin_type = OriginType.EXTRACTED_ANNOTATED
        entities_docsets = [docset for docset in self.document_set_list
                            if docset.processing_information.last_module == ENTITIES
                            # and docset.candidates.origin_type != OriginType.EXTRACTED]
                            and docset.configuration.cand_extraction_config.origin_type != OriginType.EXTRACTED]
        param_df = pd.DataFrame(columns=[TOPIC_NAME, DATASET, ORIGIN_TYPE, COREF_STRATEGY, ENTITY_METHOD, PARAMS, WORDVECTORS, TIME])

        widgets = [progressbar.FormatLabel(
            "PROGRESS: Reading details of %(value)d-th/%(max_value)d (%(percentage)d %%) entity groups (in: %(elapsed)s).")]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(entities_docsets)).start()
        for d_id, docset in enumerate(entities_docsets):
            if len(re.findall(r'[0-9]+ecbplus2', docset.topic)):
                dataset_name = "ECB_reannot"
            elif len(re.findall(r'[0-9]+ecbplus', docset.topic)):
                dataset_name = "ECB"
            elif len(re.findall(r'[0-9]+ecb', docset.topic)):
                dataset_name = "ECB"
            elif len(re.findall(r'[0-9]+nident', docset.topic)):
                dataset_name = "NIdent"
            else:
                # newswcl will have its specific annotation version
                _, dataset_name = os.path.split(docset.configuration.cand_extraction_config.annot_path)
                dataset_name = "NewsWCL_" + str(dataset_name).split("_")[0]

            param_df = param_df.append(pd.DataFrame({
                TOPIC_NAME: docset.topic,
                DATASET: dataset_name,
                ORIGIN_TYPE: docset.configuration.cand_extraction_config.origin_type.name.lower(),
                COREF_STRATEGY: docset.configuration.cand_extraction_config.coref_extraction_strategy.name.lower(),
                ENTITY_METHOD: docset.configuration.entity_method,
                PARAMS: docset.configuration.entity_identifier_config.param_source,
                WORDVECTORS: docset.configuration.entity_identifier_config.word_vectors,
                TIME: docset.processing_information.last_module_timestamp
            }, index=[d_id]))
            bar.update(d_id + 1)
        bar.finish()

        LOGGER.info("Number of docsets with duplicates: " + str(len(param_df)))
        param_df_sorted = param_df.sort_values(by=[DATASET, TOPIC_NAME, ORIGIN_TYPE, COREF_STRATEGY, ENTITY_METHOD, PARAMS, WORDVECTORS, TIME],
                             ascending=[True, True, True, True, True, True, True, True])
        param_df = param_df_sorted[list(set(param_df_sorted.columns) - {TIME})].drop_duplicates(keep="last")
        LOGGER.info("Number of docsets without duplicates: " + str(len(param_df)))

        param_df_groups = param_df.groupby([DATASET, ORIGIN_TYPE, COREF_STRATEGY, ENTITY_METHOD, PARAMS, WORDVECTORS])
        eval_results_df = pd.DataFrame()
        eval_results_type_df = pd.DataFrame()

        widgets = [progressbar.FormatLabel(
            "PROGRESS: Calculating scores of a  %(value)d-th/%(max_value)d (%(percentage)d %%) entity configurations (in: %(elapsed)s).")]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(param_df_groups)).start()

        all_datasets_df = pd.DataFrame()
        performance_df = pd.DataFrame()

        for group_id, (group_name_tuple, group_df) in enumerate(param_df_groups):
            group_name = "_".join(list(group_name_tuple))

            eval_row_df = pd.DataFrame({
                DATASET: group_name_tuple[0],
                ORIGIN_TYPE: group_name_tuple[1],
                COREF_STRATEGY: group_name_tuple[2],
                ENTITY_METHOD: group_name_tuple[3],
                PARAMS: group_name_tuple[4],
                WORDVECTORS: group_name_tuple[5]
            }, index=[group_name])

            LOGGER.info("Forming a table with data for evaluation.")
            pred_true_df, update_dataset_stats, perf_local_df = self.form_cand_tables([entities_docsets[set_id]
                                                                  for set_id in list(group_df.index)], group_name_tuple[0])

            perf_local_df[CONFIG] = [group_name] * len(perf_local_df)
            performance_df = performance_df.append(pd.merge(eval_row_df, perf_local_df.reset_index(), how="left",
                                                   left_index=True, right_on=[CONFIG]))

            if update_dataset_stats:
                dataset_stats_df, entities_stats_df = self.calc_wcl(self.datasets[group_name_tuple[0]], group_name_tuple[0])
                dataset_stats_df.index = [shortuuid.uuid(dataset_stats_df.loc[ind, DATASET] + dataset_stats_df.loc[ind, CONCEPT_TYPE])
                                          for ind in list(dataset_stats_df.index)]
                inters_ids = list(set(all_datasets_df.index).intersection(set(dataset_stats_df.index)))
                all_datasets_df.loc[inters_ids] = dataset_stats_df.loc[inters_ids]
                diff_ids = list(set(dataset_stats_df.index).difference(set(all_datasets_df.index)))
                all_datasets_df = all_datasets_df.append(dataset_stats_df.loc[diff_ids])
                self.dataset_stats[group_name_tuple[0]] = entities_stats_df

            scores_df = self.calc_entity_metrics(pred_true_df, group_name, group_name_tuple[0])
            eval_row_df_ = pd.merge(eval_row_df, scores_df, how="left", left_index=True, right_index=True)
            eval_results_df = eval_results_df.append(eval_row_df_)

            types_df = self.calc_entity_type_metrics(pred_true_df, group_name, group_name_tuple[0])
            eval_type_rows_df = pd.DataFrame({
                DATASET: [group_name_tuple[0]] * len(types_df),
                ORIGIN_TYPE: [group_name_tuple[1]] * len(types_df),
                COREF_STRATEGY: [group_name_tuple[2]] * len(types_df),
                ENTITY_METHOD: [group_name_tuple[3]] * len(types_df),
                PARAMS: [group_name_tuple[4]] * len(types_df),
                WORDVECTORS: [group_name_tuple[5]] * len(types_df)
            }, index=list(types_df.index))

            eval_results_type_df = eval_results_type_df.append(pd.merge(eval_type_rows_df, types_df, how="left",
                                                                        left_index=True, right_index=True))
            bar.update(group_id + 1)
        bar.finish()
        return [eval_results_df, eval_results_type_df, all_datasets_df, performance_df]

    def calc_wcl(self, df, dataset_name):
        results_df = pd.DataFrame(columns=[DATASET, CONCEPT_TYPE, WCL, NUMBER_ENT, NUMBER])
        local_res_df = pd.DataFrame()
        for label in set(df[LABEL].values):
            label_df = df[df[LABEL] == label]
            concept = list(set(label_df[CONCEPT_TYPE].values))[0]
            wcl_score = 0
            unique_phrases = 0
            for head in set(label_df[HEAD].values):
                head_df = label_df[label_df[HEAD] == head]
                wcl_score += len(set(head_df[WORDSET].values))/len(head_df)
                unique_phrases += len(set(head_df[WORDSET].values))
            wcl_score_final = wcl_score*unique_phrases/len(label_df) if len(label_df) > 1 else 1
            local_res_df = local_res_df.append(pd.DataFrame({
                CONCEPT_TYPE: concept,
                WCL: wcl_score_final,
                NUMBER: len(label_df)
            }, index=[label]))
            print(label, str(wcl_score_final))

        local_res_df[WCL_WEIGHTED] = local_res_df[WCL].values * local_res_df[NUMBER].values
        sum_df = local_res_df.groupby(by=[CONCEPT_TYPE])[NUMBER].sum().reset_index()
        # avg_df = local_res_df.groupby(by=[CONCEPT_TYPE])[WCL].mean().reset_index()
        w_avg_df = local_res_df.groupby(by=[CONCEPT_TYPE])[WCL_WEIGHTED].sum().reset_index()
        count_df = local_res_df.groupby(by=[CONCEPT_TYPE])[WCL].count().reset_index()
        results_df[DATASET] = [dataset_name] * len(sum_df)
        results_df[CONCEPT_TYPE] = sum_df[CONCEPT_TYPE].values
        results_df[WCL] = w_avg_df[WCL_WEIGHTED].values / sum_df[NUMBER].values
        results_df[NUMBER] = sum_df[NUMBER].values
        results_df[NUMBER_ENT] = count_df[WCL].values
        return results_df, local_res_df

    def form_cand_tables(self, selected_docsets, dataset_name):
        # full match
        df_cols = [TRUE_PHRASE, PRED_PHRASE, TRUE_LABEL, PRED_LABEL, ANNOT_TYPE, PRED_TYPE]
        pred_true_df = pd.DataFrame(columns=df_cols)
        cand_dict = {}
        # if dataset_name not in self.datasets:
        dataset_df = pd.DataFrame(columns=[LABEL, CONCEPT_TYPE, PHRASE, WORDSET, HEAD])
        # else:
        #     dataset_df = None

        perfomance_df = pd.DataFrame()

        for docsed_id, docset in enumerate(selected_docsets):

            men_num = np.sum([len(e.members) for e in docset.entities])
            time_ = 0.0

            if hasattr(docset, "entity_execution_time"):
                if type(docset.entity_execution_time) == float:
                    time_ = docset.entity_execution_time
                else:
                    time_ = docset.entity_execution_time.total_seconds() \
                        if docset.configuration.entity_method != "corenlp" else docset.cand_execution_time

            elif hasattr(docset.processing_information, "entity_execution_time"):
                if type(docset.processing_information.entity_execution_time) == float:
                    time_ = docset.processing_information.entity_execution_time
                else:
                    if hasattr(docset, "cand_execution_time"):
                        time_ = docset.cand_execution_time + \
                                docset.processing_information.entity_execution_time.total_seconds()
                    elif hasattr(docset.processing_information, "cand_execution_time"):
                        time_ = docset.processing_information.cand_execution_time + \
                                docset.processing_information.entity_execution_time.total_seconds()
                    else:
                        time_ = docset.processing_information.entity_execution_time.total_seconds()
            if time_ is not None:
                perfomance_df = perfomance_df.append(pd.DataFrame({
                    TIME: time_,
                    NUMBER: men_num,
                    AVG_TIME: time_/men_num
                }, index=[docset.topic]))

            for entity in docset.entities:
                entity_type = "action" if np.sum([m.type == CandidateType.VP for m in entity.members]) else entity.type
                for member in entity.members:
                    if member.annot_label is not None:

                        len_labels = len(member.annot_label.split("+"))
                        for l_id, label in enumerate(member.annot_label.split("+")):
                            if len_labels == 1:
                                annot_type = member.annot_type if not hasattr(member, "annot_type_full") \
                                                                else member.annot_type_full
                            else:
                                annot_types = member.annot_type.split("+")
                                if not len(annot_types):
                                    annot_type = None
                                elif not len(annot_types[-1]):
                                    annot_type = annot_types[0]
                                else:
                                    try:
                                        annot_type = annot_types[l_id]
                                    except IndexError:
                                        annot_type = annot_types[0]

                            m_id = member.id + "_" + str(l_id)
                            pred_true_df = pred_true_df.append(pd.DataFrame({
                                TRUE_PHRASE: member.annot_text,
                                PRED_PHRASE: member.text,
                                TRUE_LABEL: label + "_" + re.sub(r'\D+', "", docset.topic.split("_")[0]),
                                PRED_LABEL: entity.representative.replace(" ", "_") + "_" +
                                            entity.id + "_" + re.sub(r'\D+', "", docset.topic.split("_")[0]),
                                ANNOT_TYPE: annot_type,
                                PRED_TYPE: entity_type
                            }, index=[m_id]))
                            cand_dict[m_id] = member

                            # if dataset_df is not None:
                            dataset_df = dataset_df.append(pd.DataFrame({
                                LABEL: label + "_" + re.sub(r'\D+', "", docset.topic.split("_")[0]),
                                CONCEPT_TYPE: annot_type,
                                PHRASE: member.annot_text,
                                # WORDSET: " ".join(sorted(list(frozenset([w for w in member.annot_text.split(" ")
                                WORDSET: " ".join(sorted(list(frozenset([w for w in member.text.split(" ")
                                                                 if w not in string.punctuation
                                                                 and w not in LocalDictLists.stopwords])))),
                                HEAD: member.head_token.word
                            }, index=[m_id]))
                    else:
                        pred_true_df = pred_true_df.append(pd.DataFrame({
                            TRUE_PHRASE: member.annot_text,
                            PRED_PHRASE: member.text,
                            TRUE_LABEL: member.annot_label,
                            PRED_LABEL: entity.representative.replace(" ", "_") + "_" +
                                            entity.id + "_" + re.sub(r'\D+', "", docset.topic.split("_")[0]),
                            ANNOT_TYPE: member.annot_type,
                            PRED_TYPE: entity_type
                        }, index=[member.id]))
                        cand_dict[member.id] = member

        if len(pred_true_df) == len(pred_true_df.dropna()):
            updated_dataset = False
            if dataset_name in self.datasets:
                diff_index = set(dataset_df.index).difference(set(self.datasets[dataset_name].index))
                if len(diff_index):
                    add_df = dataset_df.loc[list(diff_index)]
                    # add_df_2 = self.datasets[dataset_name].loc[list(diff_index.intersection(set(self.datasets[dataset_name].index)))]
                    self.datasets[dataset_name] = self.datasets[dataset_name].append(add_df)
                    updated_dataset = True
            else:
                self.datasets[dataset_name] = dataset_df
                updated_dataset = True
            # full match of annot to pred
            return pred_true_df, updated_dataset, perfomance_df

        # fuzzy match
        pred_true_filled_df = pd.DataFrame(columns=df_cols)
        for label_pred in set(pred_true_df[PRED_LABEL].values):
            # if an identical phrase to an annotated is in the same entity, label it too
            local_df = pred_true_df[pred_true_df[PRED_LABEL] == label_pred]
            local_df_nona = local_df.dropna()

            if len(local_df_nona) != len(local_df):
                phrase_df = local_df_nona[[TRUE_PHRASE, PRED_PHRASE, TRUE_LABEL, ANNOT_TYPE]].drop_duplicates(keep="first")
                phrase_df[TRUE_PHRASE] = [v.lower() for v in phrase_df[TRUE_PHRASE].values]
                phrase_df[PRED_PHRASE] = [v.lower() for v in phrase_df[PRED_PHRASE].values]
                phrase_df.reset_index(inplace=True)
                phrase_df.set_index([TRUE_PHRASE], inplace=True)

                for index_local, row_local in local_df.iterrows():
                    if row_local[TRUE_PHRASE] is not None:
                        continue

                    for index_phrase, row_phrase in phrase_df.iterrows():
                        cand_pred = cand_dict[index_local]
                        cand_true = cand_dict[row_phrase["index"]]
                        cand_pred_appos = ""
                        for dep in cand_pred.dependency_subtree:
                            if dep.dep == "appos" and dep.governor == cand_pred.head_token.index:
                                cand_pred_appos = dep.dependent_gloss
                                break

                        cand_true_appos = ""
                        for dep in cand_true.dependency_subtree:
                            if dep.dep == "appos" and dep.governor == cand_true.head_token.index:
                                cand_true_appos = dep.dependent_gloss
                                break

                        # check on heads and appos
                        if cand_pred.head_token.word.lower() == cand_true.head_token.word.lower() or \
                            cand_pred.head_token.word.lower() == cand_true_appos.lower() or \
                                cand_true.head_token.word.lower() == cand_pred_appos.lower():
                            local_df.loc[index_local, TRUE_PHRASE] = index_phrase
                            local_df.loc[index_local, TRUE_LABEL] = row_phrase[TRUE_LABEL]
                            local_df.loc[index_local, ANNOT_TYPE] = row_phrase[ANNOT_TYPE]
                            break

            pred_true_filled_df = pred_true_filled_df.append(local_df)
        # return pred_true_filled_df.dropna()
        self.datasets[dataset_name] = dataset_df
        return pred_true_filled_df.fillna(UNK), False, perfomance_df

    def calc_entity_metrics(self, pred_true_df, index, dataset_name):
        true_dict = {}
        pred_dict = {}

        for label in set(pred_true_df[TRUE_LABEL].values):
            if label == UNK:
                continue
            true_dict[label] = list(pred_true_df[pred_true_df[TRUE_LABEL] == label].index)

        for label in set(pred_true_df[PRED_LABEL].values):
            pred_dict[label] = list(pred_true_df[pred_true_df[PRED_LABEL] == label].index)

        metrics_df = pd.DataFrame()

        metrics = {
            MUC: self._muc,
            B3: self._b3,
            # CEAF_M: self._ceaf_m,
            CEAF_E: self._ceaf_e,
            # AVG: self._f1_micro,
            WEIGHTED: self._weighted_f1
            # ,
            # WCL_WEIGHTED: self._weighted_f1_wcl
        }
        avg_f1 = []

        conll_count = 0

        for metrics_name, metrics_func in metrics.items():
            LOGGER.info("Calculating {} score.".format(metrics_name))
            p, r, f1 = metrics_func(true_dict, pred_dict, dataset_name)
            metrics_df = pd.concat([metrics_df, pd.DataFrame({
                P + metrics_name: p,
                R + metrics_name: r,
                F1 + metrics_name: f1
            }, index=[index])], axis=1, sort=False)

            if metrics_name in [MUC, B3, CEAF_E]:
                avg_f1.append(f1)
                conll_count += 1

            if conll_count == 3:
                metrics_df = pd.concat([metrics_df, pd.DataFrame({
                    F1 + CONLL: np.mean(avg_f1)
                }, index=[index])], axis=1, sort=False)
                conll_count = 0

        # true_pred_df = pred_true_df[pred_true_df[TRUE_LABEL] != UNK][[TRUE_LABEL, PRED_LABEL]]
        # h,c,v = hcv(true_pred_df[TRUE_LABEL].values, true_pred_df[PRED_LABEL].values)
        # metrics_df = pd.concat([metrics_df, pd.DataFrame({
        #     H: h,
        #     C: c,
        #     V: v
        # }, index=[index])], axis=1, sort=False)

        return metrics_df

    def calc_entity_type_metrics(self, pred_true_df, index, dataset_name):
        metrics_df = pd.DataFrame()

        for ent_type in sorted(list(set(pred_true_df[ANNOT_TYPE]))):
            if ent_type == UNK:
                continue
            LOGGER.info("Calculating scores for {} type".format(ent_type))
            type_df = pred_true_df[pred_true_df[ANNOT_TYPE] == ent_type]
            metrics_local_df = pd.DataFrame()

            true_dict = {}
            pred_dict = {}

            for label in set(type_df[TRUE_LABEL].values):
                true_dict[label] = list(type_df[type_df[TRUE_LABEL] == label].index)

            for label in set(type_df[PRED_LABEL].values):
                pred_dict[label] = list(pred_true_df[pred_true_df[PRED_LABEL] == label].index)

            row_index = shortuuid.uuid(index + ent_type)

            metrics = {
                MUC: self._muc,
                B3: self._b3,
                # CEAF_M: self._ceaf_m,
                CEAF_E: self._ceaf_e,
                # ,
                # AVG: self._f1_micro,
                WEIGHTED: self._weighted_f1
                # ,
                # WCL_WEIGHTED: self._weighted_f1_wcl
            }
            avg_f1 = []

            metrics_local_df = pd.concat([metrics_local_df, pd.DataFrame({
                # CONFIG: index,
                ANNOT_TYPE: ent_type
            }, index=[row_index])])

            conll_count = 0
            for metrics_name, metrics_func in metrics.items():
                # LOGGER.info("Calculating {} score.".format(metrics_name))
                p, r, f1 = metrics_func(true_dict, pred_dict, dataset_name)
                metrics_local_df = pd.concat([metrics_local_df, pd.DataFrame({
                    P + metrics_name: p,
                    R + metrics_name: r,
                    F1 + metrics_name: f1
                }, index=[row_index])], axis=1, sort=False)
                # avg_f1.append(f1)

                if metrics_name in [MUC, B3, CEAF_E]:
                    avg_f1.append(f1)
                    conll_count += 1

                if conll_count == 3:
                    metrics_local_df = pd.concat([metrics_local_df, pd.DataFrame({
                        F1 + CONLL: np.mean(avg_f1)
                    }, index=[row_index])], axis=1, sort=False)
                    conll_count = 0

            # metrics_local_df = pd.concat([metrics_local_df, pd.DataFrame({
            #     F1 + CONLL: np.mean(avg_f1[:3])
            # }, index=[row_index])], axis=1, sort=False)

            metrics_local_df = pd.concat([metrics_local_df, pd.DataFrame({
                SUPPORT: len(type_df)
            }, index=[row_index])], axis=1, sort=False)

            # true_pred_df = type_df[type_df[TRUE_LABEL] != UNK][[TRUE_LABEL, PRED_LABEL]]
            # h,c,v = hcv(true_pred_df[TRUE_LABEL].values, true_pred_df[PRED_LABEL].values)
            # metrics_local_df = pd.concat([metrics_local_df, pd.DataFrame({
            #     H: h,
            #     C: c,
            #     V: v
            # }, index=[row_index])], axis=1, sort=False)

            metrics_df = metrics_df.append(metrics_local_df)

        return metrics_df

    @staticmethod
    def f1_score(p, r):
        return 2*r*p/(p+r)

    def _muc(self, true_dict, pred_dict, dataset_name=None):

        # recall
        sum_nom_r = 0
        sum_denom_r = 0
        for true_label, true_cands in true_dict.items():
            if not len(true_cands):
                continue
            partitions = []

            for pred_label, pred_cands in pred_dict.items():
                inters = set(true_cands).intersection(set(pred_cands))
                if len(inters):
                    # we partition a predicted chain
                    partitions.append(inters)

            # remaining unresolved mentions in a true entity form an extra partition
            partitions_union = set().union(*partitions)
            if len(true_cands) != len(partitions_union):
                partitions.append(set(true_cands).difference(partitions_union))

            sum_nom_r += len(true_cands) - len(partitions) if len(true_cands) > 1 else 1
            sum_denom_r += len(true_cands) - 1 if len(true_cands) > 1 else 1
        recall = sum_nom_r / sum_denom_r

        # precision
        sum_nom_p = 0
        sum_denom_p = 0
        for pred_label, pred_cands in pred_dict.items():
            if not len(pred_cands):
                continue
            partitions = []

            for true_label, true_cands in true_dict.items():
                inters = set(true_cands).intersection(set(pred_cands))
                if len(inters):
                    partitions.append(inters)

            partitions_union = set().union(*partitions)
            if len(pred_cands) != len(partitions_union):
                partitions.append(set(pred_cands).difference(partitions_union))
                # for v in set(pred_cands).difference(partitions_union):
                #     partitions.append([v])

            sum_nom_p += len(pred_cands) - len(partitions) if len(pred_cands) > 1 else 1
            sum_denom_p += len(pred_cands) - 1 if len(pred_cands) > 1 else 1

        precision = sum_nom_p / sum_denom_p
        return precision, recall, EvalScorer.f1_score(precision, recall) if precision > 0 and recall > 0 else 0

    def _b3(self, true_dict, pred_dict, dataset_name=None):
        # recall
        sum_nom_r = 0
        # precision
        sum_nom_p = 0

        for true_label, true_cands in true_dict.items():

            for pred_label, pred_cands in pred_dict.items():
                inters = set(true_cands).intersection(set(pred_cands))
                if not len(inters):
                    continue

                sqr = len(inters)**2
                sum_nom_r += sqr/len(true_cands)
                sum_nom_p += sqr/len(pred_cands)
        sum_denom_p = np.sum([len(v) for v in list(pred_dict.values())])
        sum_denom_r = np.sum([len(v) for v in list(true_dict.values())])
        recall = sum_nom_r / sum_denom_r
        precision = sum_nom_p / sum_denom_p
        return precision, recall, EvalScorer.f1_score(precision, recall) if precision > 0 and recall > 0 else 0

    def _ceaf_e(self, true_dict, pred_dict, dataset_name=None):
        true_labels_dict = {}
        for true_label, true_cands in true_dict.items():
            topic_id = true_label.split("_")[-1]
            if topic_id not in true_labels_dict:
                true_labels_dict[topic_id] = {}
            true_labels_dict[topic_id].update({true_label: true_cands})

        pred_label_dict = {}
        for pred_label, pred_cands in pred_dict.items():
            topic_id = pred_label.split("_")[-1]
            if topic_id not in pred_label_dict:
                pred_label_dict[topic_id] = {}
            pred_label_dict[topic_id].update({pred_label: pred_cands})

        sum_nom_r = 0
        widgets = [progressbar.FormatLabel(
            "PROGRESS: Calculating ceaf_e for %(value)d-th/%(max_value)d (%(percentage)d %%) topic (in: %(elapsed)s).")]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(true_labels_dict)).start()

        for i, (topic_id, true_dict_) in enumerate(true_labels_dict.items()):
            pred_dict_ = pred_label_dict[topic_id]

            matrix = []
            for true_label, true_cands in true_dict_.items():
                array = []
                for pred_label, pred_cands in pred_dict_.items():
                    inters = set(true_cands).intersection(set(pred_cands))
                    array.append(len(inters))
                matrix.append(array)

            cost_matrix = make_cost_matrix(matrix)
            m = Munkres()
            indexes = m.compute(cost_matrix)
            # indexes = [(i, np.argmax(row)) for i, row in enumerate(matrix)]

            for row, column in indexes:
                if matrix[row][column] == 0:
                    continue
                sum_nom_r += 2 * matrix[row][column] / (len(list(true_dict_.values())[row]) + len(list(pred_dict_.values())[column]))
            bar.update(i + 1)
        bar.finish()

        recall = sum_nom_r / len(true_dict)
        # more_1 =  [v for v in list(pred_dict.values()) if len(v) > 1]
        # if len(more_1):
        #     precision = sum_nom_r / len(more_1)
        # else:
        precision = sum_nom_r / len(pred_dict)
        return precision, recall, EvalScorer.f1_score(precision, recall) if precision > 0 and recall > 0 else 0

    def _f1_micro(self, true_dict, pred_dict, dataset_name=None):
        # recall & precision
        sum_nom = []
        sum_denom_r = []
        sum_denom_p = []
        sum_all = []

        pred_dict_sorted = {k:v for k,v in sorted(pred_dict.items(), reverse=False, key=lambda x: len(x[1]))}

        for true_label, true_cands in true_dict.items():
            partitions = []
            len_pred = []
            for pred_label, pred_cands in pred_dict_sorted.items():
                inters = set(true_cands).intersection(set(pred_cands))
                # if len(inters):
                partitions.append(inters)
                len_pred.append(len(pred_cands))
            a_max = int(np.argmax([len(v) for v in partitions]))
            sum_denom_p.append(len_pred[a_max])
            sum_nom.append(len(partitions[a_max])) # if len(partitions) else 0)
            sum_denom_r.append(len(true_cands))
            sum_all.append()
        # recall = np.mean([sum_nom[v] / sum_denom_r[v] for v in range(len(sum_nom))])
        # precision = np.mean([sum_nom[v] / sum_denom_p[v] for v in range(len(sum_nom))])
        # f1 = np.mean([EvalScorer.f1_score(sum_nom[v] / sum_denom_p[v], sum_nom[v] / sum_denom_r[v])
        #              for v in range(len(sum_nom))])
        recall = np.sum(sum_nom)/ np.sum(sum_denom_r)
        precision = np.sum(sum_nom) / np.sum(sum_denom_p)
        f1 = np.sum(sum_nom) / np.sum(sum_denom_r)
        return precision, recall, f1

    def _weighted_f1(self, true_dict, pred_dict, dataset_name=None):
        # recall & precision
        sum_nom = []
        sum_denom_r = []
        sum_denom_p = []
        weights = []

        pred_dict_sorted = {k:v for k,v in sorted(pred_dict.items(), reverse=False, key=lambda x: len(x[1]))}

        for true_label, true_cands in true_dict.items():
            partitions = []
            len_pred = []
            for pred_label, pred_cands in pred_dict_sorted.items():
                inters = set(true_cands).intersection(set(pred_cands))
                # if len(inters):
                partitions.append(inters)
                len_pred.append(len(pred_cands))
            a_max = int(np.argmax([len(v) for v in partitions]))
            sum_denom_p.append(len_pred[a_max])
            sum_nom.append(len(partitions[a_max])) # if len(partitions) else 0)
            sum_denom_r.append(len(true_cands))
            weights.append(len(true_cands))
        recall = np.sum([(sum_nom[v] / sum_denom_r[v]) * weights[v] for v in range(len(sum_nom))]) / np.sum(weights)
        precision = np.sum([(sum_nom[v] / sum_denom_p[v]) * weights[v] for v in range(len(sum_nom))]) / np.sum(weights)
        f1 = np.sum([EvalScorer.f1_score(sum_nom[v] / sum_denom_p[v], sum_nom[v] / sum_denom_r[v]) * weights[v]
                     for v in range(len(sum_nom))]) / np.sum(weights)
        return precision, recall, f1

    def _weighted_f1_wcl(self, true_dict, pred_dict, dataset_name):
        # recall & precision
        sum_nom = []
        sum_denom_r = []
        sum_denom_p = []
        weights = []

        pred_dict_sorted = {k: v for k, v in sorted(pred_dict.items(), reverse=False, key=lambda x: len(x[1]))}

        for true_label, true_cands in true_dict.items():
            partitions = []
            len_pred = []
            for pred_label, pred_cands in pred_dict_sorted.items():
                inters = set(true_cands).intersection(set(pred_cands))
                # if len(inters):
                partitions.append(inters)
                len_pred.append(len(pred_cands))
            a_max = int(np.argmax([len(v) for v in partitions]))
            sum_denom_p.append(len_pred[a_max])
            sum_nom.append(len(partitions[a_max]))
            sum_denom_r.append(len(true_cands))
            weights.append(self.dataset_stats[dataset_name].loc[true_label, WCL])
        recall = np.sum((sum_nom[v] / sum_denom_r [v]) * weights[v] for v in range(len(sum_nom))) / np.sum(weights)
        precision = np.sum([(sum_nom[v] / sum_denom_p[v]) * weights[v] for v in range(len(sum_nom))]) / np.sum(weights)
        f1 = np.sum([EvalScorer.f1_score(sum_nom[v] / sum_denom_p[v], sum_nom[v] / sum_denom_r[v]) * weights[v]
                     for v in range(len(sum_nom))]) / np.sum(weights)
        return precision, recall, f1

    # def _blanc(self, true_dict, pred_dict):
    #     pass
