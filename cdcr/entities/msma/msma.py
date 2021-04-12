from cdcr.entities.msma.entity_preprocessor_msma import EntityPreprocessorMSMA
from cdcr.structures.entity_set import EntitySet
from cdcr.entities.msma.entity_msma import EntityMSMA
from cdcr.config import *
from cdcr.entities.identifier import Identifier
from cdcr.util.magnitude_wordvectors import MagnitudeModel
from cdcr.entities.const_dict_global import *

import os
import nltk
import warnings
import pickle
import pandas as pd
import progressbar


warnings.filterwarnings('ignore')
nltk.download('wordnet')

CAND, HEADWORD, HEAD_ID, SENT_ID, DOC_ID, ANNOT_LABEL, ANNOT_TYPE, ENT_NAME, ENT_TYPE, ENT_SIZE = "phrase", "headword", "head_id", \
                                            "sent_id", "doc_id", "annot_label", "annot_category", "_entname", "_type", "_size"
MESSAGES = {
    "ent_created": "Created {0} entities. \n",
    "round": "Round {0}: Merging entities using {1}:",
    "merge_res": "Reduced the number of entities from {0} to {1}. \n",
    "eval_mode": "Data collection for evaluation is on.",
    "eval_process": "PROGRESS: Saving details of %(value)d-th/%(max_value)d (%(percentage)d %%) entities (in: %(elapsed)s)."
}
INIT_RUN = True


class MultiStepMergingApproach(Identifier):
    """
    A multistep merging approach (MSMA) identifies searches for similar entities by appliying different similarity
    criteria at each merge step and merge entities together if they are identified as similar.

    MSMA exists in two versions, where we experimented with different variations of merge steps and ways how to
    process entities.
    """

    logger = LOGGER

    def __init__(self, docs, entity_class=EntityMSMA, ent_preprocessor=EntityPreprocessorMSMA):

        super().__init__(docs)

        self.config = docs.configuration.entity_identifier_config.params

        # self.model = None
        self.model = MagnitudeModel(docs.configuration.entity_identifier_config.word_vectors)
        self.entity_dict = {}

        # steps and entity need to be defined for a specific MSMA implementation
        self.steps = {}
        self.entity_class = entity_class

        self.ent_preprocessor = ent_preprocessor(self.docs, self.entity_class)
        self.eval_df = None

    def extract_entities(self):

        # ENTITY PREPROCESSING
        # self.ent_preprocessor = EntityPreprocessorMSMA(self.docs, self.entity_class)
        if INIT_RUN:
            if self.docs.configuration.entity_identifier_config.load_preproc:
                if any([file for file in os.listdir(TMP_PATH) if self.docs.topic.split("_")[0] in file and XCOREF in file]):#self.docs.configuration.entity_method in file]):
                    self.load_interm_results()

            if not len(self.entity_dict):
                self.entity_dict = self.ent_preprocessor.entity_dict_construction()
                self.cache_interm_results()
        else:
            self.load_interm_results()
        self.entity_dict = {key: value for (key, value) in sorted(self.entity_dict.items(), reverse=True,
                                                                  key=lambda x: len(x[1].members))}
        initial_entity_num = len(self.entity_dict)
        self.logger.info(MESSAGES["ent_created"].format(initial_entity_num))

        # self.model = keep_word2vec.load(WORD2VEC_MODEL_PATH)

        # EVALUATION SETUP
        if self.config.evaluation.evaluation_mode:
            self.logger.info(MESSAGES["eval_mode"])
            self.init_eval_table()

        # RUN MERGE STEPS #
        for step_code, step_function in self.steps.items():

            if step_code not in self.config.steps_to_execute:
                continue

            init_len = len(self.entity_dict)
            # if step_code in [STEP0, STEP1]:
            #     continue

            step = step_function(step_code, self.docs, self.entity_dict, self.ent_preprocessor, self.model)
            self.entity_dict = step.merge()
            # self.cache_interm_results()

            end_len = len(self.entity_dict)
            self.logger.info(MESSAGES["merge_res"].format(init_len, end_len))

            if self.config.evaluation.evaluation_mode:
                self.update_eval_table(step_code)

        entity_set = EntitySet(self.docs.configuration.entity_method, self.docs.topic, list(self.entity_dict.values()),
                               evaluation_details=self.eval_df)
        return entity_set

    def init_eval_table(self):
        """
        Creates a table to evaluate merge steps.
        """
        self.eval_df = pd.DataFrame()

        widgets = [progressbar.FormatLabel(MESSAGES["eval_process"])]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(self.entity_dict)).start()

        for i, (key, ent) in enumerate(self.entity_dict.items()):
            for m in ent.members:
                self.eval_df = self.eval_df.append(pd.DataFrame({
                    CAND: m.text,
                    HEADWORD: m.head_token.word,
                    HEAD_ID: m.head_token.index,
                    SENT_ID: m.sentence.index,
                    DOC_ID: m.document.id,
                    ANNOT_LABEL: m.annot_label,
                    ANNOT_TYPE: m.annot_type,
                    INIT_STEP + ENT_NAME: ent.name,
                    INIT_STEP + ENT_TYPE: ent.type,
                    INIT_STEP + ENT_SIZE: len(ent.members)
                }, index=[m.id]))
            bar.update(i+1)
        bar.finish()

    def update_eval_table(self, step_name: str):
        """
        Updates a table to evaluate merge steps.
        """
        widgets = [progressbar.FormatLabel(MESSAGES["eval_process"])]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(self.entity_dict)).start()
        new_res_df = pd.DataFrame()
        for i, (key, ent) in enumerate(self.entity_dict.items()):
            for m in ent.members:
                new_res_df = new_res_df.append(pd.DataFrame({
                    step_name + ENT_NAME: ent.name,
                    step_name + ENT_TYPE: ent.type,
                    step_name + ENT_SIZE: len(ent.members)
                }, index=[m.id]))
            bar.update(i+1)
        bar.finish()
        self.eval_df = pd.merge(self.eval_df, new_res_df, how="left", left_index=True, right_index=True)
        self.eval_df.sort_values(by=[step_name + ENT_SIZE, step_name + ENT_NAME], ascending=[False, False],
                                                                                                      inplace=True)

    def cache_interm_results(self):
        """
        Cache results of entity preprocessing for faster restoring when debugging.

        """
        topic_id = self.docs.topic.split("_")[0]
        approach = self.docs.configuration.entity_method

        for k, v in self.ent_preprocessor.__dict__.items():
            if type(v) is dict:
                with open(os.path.join(TMP_PATH, k + "_" + topic_id + "_" + approach + ".pickle"), "wb") as file:
                    pickle.dump(getattr(self.ent_preprocessor, k), file)

        for k, v in self.__dict__.items():
            if k in ["entity_dict"]:
                with open(os.path.join(TMP_PATH, k + "_" + topic_id + "_" + approach + ".pickle"), "wb") as file:
                    pickle.dump(getattr(self, k), file)

    def load_interm_results(self):
        """
        Loads cached results for each topic. To use the function, comment L62, L63 and uncomment L64.

        """
        topic_id = self.docs.topic.split("_")[0]
        approach = self.docs.configuration.entity_method if self.docs.configuration.entity_method != XCOREF_HC else XCOREF

        for k, v in self.ent_preprocessor.__dict__.items():
            file_name = k + "_" + topic_id + "_" + approach + ".pickle"
            if file_name not in os.listdir(TMP_PATH):
                continue

            with open(os.path.join(TMP_PATH,file_name), "rb") as file:
                setattr(self.ent_preprocessor, k, pickle.load(file))

        for k, v in self.__dict__.items():
            if k not in ["entity_dict"]:
                continue

            file_name = k + "_" + topic_id + "_" + approach + ".pickle"
            if file_name not in os.listdir(TMP_PATH):
                continue

            with open(os.path.join(TMP_PATH, file_name), "rb") as file:
                setattr(self, k, pickle.load(file))
