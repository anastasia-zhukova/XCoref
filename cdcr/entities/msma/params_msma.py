import cdcr.config as config
from cdcr.entities.params_entities import *
from typing import Dict

import logging
import os
import pandas as pd
import json


MESSAGES = {"No_param": "No param {0} from a config json file was found in the default config. "
                                            "The default values of this parameter will be used. ",
            "no_key": "No key {0} or column {1} from a json config file found in the comparison table {2} layout."}

COMP_TABLES_NAME = "comparison_tables"
CONFIG_PARAMS_ENTITY_IDENTIFICATION_NAME = "config_params_entity_identification"


class MSMAParams(ParamsEntities):
    """
    A general class of parameters for multi-step-merging approach.
    Table values and parameters are different in each MSMA implementation.
    """

    tables = None

    def get_default_params(self, def_folder_path: str):
        if self.word_vectors == "not_specified":
            self.word_vectors = self.get_wordvectors_type()
        self.params = self.add_params(self.create_init_params())
        self.tables = MSMAParams.read_comp_tables(os.path.join(config.NEWALYZE_PATH, def_folder_path, COMP_TABLES_NAME))
        return self

    def create_init_params(self):
        return self.params.__dict__

    def add_params(self, params: Dict[str, Params]) -> Params:
        raise NotImplementedError

    def get_wordvectors_type(self) -> str:
        raise NotImplementedError

    @staticmethod
    def read_comp_tables(folder_path):
        tables = {}
        for file_name in os.listdir(folder_path):
            tables[file_name.split(".")[0]] = pd.read_csv(os.path.join(folder_path, file_name),
                           index_col=0)
        return tables

    def read_config(self, config_path):
        logger = logging.getLogger(self.__class__.__qualname__)
        # read configured tables
        user_tables = MSMAParams.read_comp_tables(os.path.join(config_path, COMP_TABLES_NAME))
        for table_key, table in user_tables.items():
            if table_key not in self.tables:
                continue
            for index in list(table.index):
                for column in list(table.columns):
                    try:
                        self.tables[table_key].loc[index, column] = table.loc[index, column]
                    except KeyError:
                        logger.warning(MESSAGES["no_key"].format(index, column, table_key))

        # read json files
        with open(os.path.join(config_path, CONFIG_PARAMS_ENTITY_IDENTIFICATION_NAME + ".json")) as run_config:
            json_file = json.load(run_config)

        for step_key, step_dict in json_file.items():
            try:
                if type(step_dict) != dict:
                    # getattr(self.params, step_key)
                    setattr(self.params, step_key, step_dict)
                    continue
                step = getattr(self.params, step_key)
                for k, v in step.__dict__.items():
                    # getattr(step, k)
                    setattr(step, k, step_dict[k])
            except AttributeError as e:
                logger.warning(e)
            except KeyError as e:
                logger.warning(e)
                continue
        # print()
        self.param_source = "custom"

    def save_config(self, config_path):
        tables_path = os.path.join(config_path, COMP_TABLES_NAME)
        os.mkdir(tables_path)
        for key, table in self.tables.items():
            table.to_csv(os.path.join(tables_path, key + ".csv"))

        with open(os.path.join(config_path, CONFIG_PARAMS_ENTITY_IDENTIFICATION_NAME + ".json"), "w") as file:
            json.dump({k_param: param if type(param) == list else param.__dict__
                       for k_param, param in self.params.__dict__.items()}, file)
        print()
