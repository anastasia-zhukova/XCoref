from cdcr.entities.params_entities import *
from cdcr.config import *
import os
import re


CONFIG_NAME = "config_eecdcr"
ELMO = "ELMO_Original_55B"
BERT_SRL = "BERT_SRL"
GOLD_DATA_PATH = "data/gold/cybulska_gold"
# MODELS = "models/cybulska_setup"
MODELS = "eecdcr_models"


class ParamsEECDCR(ParamsEntities):
    """
    A class with parametes required for execution of EECDCR entity identifer.
    """

    def get_default_params(self, def_folder_path=None):
        rel_path_eecdcr, _ = os.path.split(os.path.relpath(__file__))
        abs_path_eecdcr, _ = os.path.split(os.path.abspath(__file__))

        params = {
            "cd_event_model_path": os.path.join(RESOURCES_PATH, MODELS, "cd_event_best_model"),
            "cd_entity_model_path": os.path.join(RESOURCES_PATH, MODELS, "cd_entity_best_model"),
            "wd_entity_coref_file": os.path.join(rel_path_eecdcr,
                                                 "data/external/stanford_neural_wd_entity_coref_out/ecb_wd_coref.json"),
            "event_gold_file_path": os.path.join(rel_path_eecdcr, GOLD_DATA_PATH,
                                                                      "CD_test_event_mention_based.key_conll"),
            "entity_gold_file_path": os.path.join(rel_path_eecdcr, GOLD_DATA_PATH,
                                                                     "CD_test_entity_mention_based.key_conll"),
            "options_file": os.path.join(WORDVECTOR_MODELS_PATH, ELMO,
                                                             "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"),
            "weight_file": os.path.join(WORDVECTOR_MODELS_PATH, ELMO,
                                                             "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"),
            "bert_file": os.path.join(WORDVECTOR_MODELS_PATH, BERT_SRL, "bert-base-srl-2019.06.17.tar.gz"),
            "predicted_topics_path": os.path.join(rel_path_eecdcr,
                                                            "data/external/document_clustering/predicted_topics"),
            "gpu_num": -1,
            "event_merge_threshold": 0.5,
            "entity_merge_threshold": 0.5,
            "use_args_feats": True,
            "use_binary_feats": True,
            "test_use_gold_mentions": True,
            "merge_iters": 2,
            "load_predicted_topics": False,
            "seed": 1,
            "random_seed": 2048,
            "use_elmo": True,
            "use_dep": True,
            "use_srl": True,
            "use_left_right_mentions": True,
            "load_predicted_mentions": False,
            "load_elmo": True
        }
        self.params.update(params)
        self.word_vectors = GLOVE_WE

        return self
