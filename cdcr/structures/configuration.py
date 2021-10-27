"""Configuration for a document set."""
import datetime
from enum import Enum
import copy

from cdcr.candidates.params_cand import ParamsCand
from cdcr.entities.const_dict_global import *
from cdcr.entities.sieve_based.tca_improved.params_tca import ParamsTCA
from cdcr.entities.eecdcr.params_eecdcr import ParamsEECDCR
from cdcr.entities.sieve_based.xcoref.params_xcoref import ParamsXCoref
from cdcr.entities.sieve_based.xcoref_base.params_xcoref_base import ParamsXCorefBase
from cdcr.entities.multidoc_corenlp.params_corenlp import ParamsCoreNLP
from cdcr.entities.clustering.params_clustering import ParamsClustering
from cdcr.entities.lemma.params_lemmas import ParamsLemmas
from cdcr.config import *
from cdcr.structures.params import *
from cdcr.logger import LOGGER
from cdcr.entities.params_entities import ParamsEntities

MESSAGES = {
    "available_folders": "\nThe following saved configurations found: ",
    "one_folder": "Run configuration will be restored from the only folder with saved configuration.",
    "select_config": "\nSelect folder id with the run configuration. Type anything to run with the default "
                     "parameters: ",
    "no_config": "\nNo saved run configuration for the {} for the topic {} found. ",
    "def_config": "The pipeline will be executed with default parameters. \n",
    "methods": "\nChoose a method for ENTITY IDENTIFICATION: ",
    "sel_method": "\nSelect id of the option: ",
    "no_entity_method": "No such method found. The pipeline will be executed with a default method ({0}). ",
    "no_param_config": "The method will be executed with default parameters. ",
    "save_def_config": "\nDo you want to save the default config for {0} as files for later modification? (y, n) ",
    "cand_methods": "\nChoose default or saved parameters for CANDIDATE EXTRACTION:",
    "vis_methods": "\nChoose a VISUALIZATION: ",
    "user_interf": "\nDo you want to run with default config with default settings? (y, n) \nIf \"n\", you will be "
                   "offered to select and read saved config. ",
    "cand_settings": "Candidate extraction module has the following settings: ",
    "ent_settings": "Entity identification module has the following settings: ",
    "vis_settings": "Visualization module has the following settings: ",
    "msma1_recom": "If available, we recommend to use saved custom settings for TCA_IMPROVED to achieve better performance "
                   "results. To do so, restart the pipeline and select \"n\" in the default setting selection and then "
                   "choose saved config for TCA_IMPROVED. \n",
    "saved": "Configuration is saved to {}.",
    "no_req_config_id": "No requested saved method config found. The last config will be executed."
}

ENTITIES = "entities"
CANDIDATES = "candidates"
VISUALIZATION = "visualization"
DEFAULT_CAND_METHOD_NAME = "default_candidate_settings"
REALWORD_CAND_METHOD_NAME = "real_word_settings"
ANNOT_CAND_METHOD_NAME = "annot_settings"
CUSTOM_CAND_METHOD_NAME = "custom_candidate_settings"


# ----- CHANGE THESE LINES FOR FAST OPTION SELECTION WHILE DEBUGGING -----#
DEFAULT_ENTITY_METHOD = XCOREF
DEFAULT_CAND_METHOD = DEFAULT_CAND_METHOD_NAME
REALWORD_CAND_METHOD = REALWORD_CAND_METHOD_NAME
ANNOT_CAND_METHOD = ANNOT_CAND_METHOD_NAME
# ----- ------------------------------------------------------------ -----#

NOW = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


class Configuration:
    """
    A class with custom configuration of newsalyze. Enables both execution with default setup and selection of custom
    parametes in a dialog form.
    """
    def __init__(self, document_set, config=None):
        self.topic = document_set.topic

        defaul_config = {
                CANDIDATES: {
                    TCA_ORIG: ParamsCand(self.topic).get_default_params(),
                    TCA_IMPROVED: ParamsCand(self.topic).get_default_params(),
                    XCOREF:  ParamsCand(self.topic).get_default_params(),
                    XCOREF_BASE: ParamsCand(self.topic).get_default_params(),
                    EECDCR: ParamsCand(self.topic).get_barhom_params(),
                    CORENLP: ParamsCand(self.topic).get_corenlp_params(),
                    CLUSTERING: ParamsCand(self.topic).get_lemma_params(),
                    LEMMA: ParamsCand(self.topic).get_lemma_params(),
                    ORIG_ANNOT: ParamsCand(self.topic).get_lemma_params(),

                    DEFAULT_CAND_METHOD_NAME: ParamsCand(self.topic).get_default_params(),
                    REALWORD_CAND_METHOD_NAME: ParamsCand(self.topic).get_realword_setup(),
                    ANNOT_CAND_METHOD_NAME: ParamsCand(self.topic).get_annot_setup(),
                    CUSTOM_CAND_METHOD_NAME: None
                },
                ENTITIES: {
                    TCA_ORIG: ParamsTCA(),
                    TCA_IMPROVED: ParamsTCA(),
                    XCOREF: ParamsXCoref(),
                    XCOREF_BASE: ParamsXCorefBase(),
                    EECDCR: ParamsEECDCR(),
                    CORENLP: ParamsCoreNLP(),
                    CLUSTERING: ParamsClustering(),
                    LEMMA: ParamsLemmas(),
                    ORIG_ANNOT: ParamsEntities()
                }
            }

        self.def_params = {
            TCA_ORIG: os.path.join(ENTITIES, SIEVE_BASED, TCA_ORIG),
            TCA_IMPROVED: os.path.join(ENTITIES, SIEVE_BASED, TCA_IMPROVED),
            XCOREF: os.path.join(ENTITIES, SIEVE_BASED, XCOREF),
            XCOREF_BASE: os.path.join(ENTITIES, SIEVE_BASED, XCOREF_BASE),
            EECDCR: os.path.join(ENTITIES, EECDCR),
            CORENLP: os.path.join(ENTITIES, CORENLP),
            CLUSTERING: os.path.join(ENTITIES, CLUSTERING),
            LEMMA: os.path.join(ENTITIES, LEMMA),
            ORIG_ANNOT: os.path.join(ENTITIES, ORIG_ANNOT)
        }

        if not hasattr(document_set, "configuration"):
            self._run_config = defaul_config

            self.entity_method = DEFAULT_ENTITY_METHOD
            self.candidate_method = DEFAULT_CAND_METHOD
        else:
            docset_config = copy.deepcopy(document_set.configuration._run_config)
            for module, module_params in defaul_config.items():
                for method, method_config in module_params.items():
                    if method not in docset_config[module]:
                        docset_config[module][method] = method_config
                    if defaul_config[module][method] is not None:
                        for param in list(defaul_config[module][method].__dict__):
                            if param not in docset_config[module][method].__dict__:
                                setattr(docset_config[module][method], param, defaul_config[module][method].__dict__[param])
            self._run_config = docset_config
            self.entity_method = document_set.configuration.entity_method
            self.candidate_method = document_set.configuration.candidate_method
            for cand_method in list(self._run_config[CANDIDATES]):
                if self._run_config[CANDIDATES][cand_method] is None:
                    continue
                self._run_config[CANDIDATES][cand_method].read_annot(self.topic,
                                                        self._run_config[CANDIDATES][cand_method].annot_index)

        # entities need to be before candidates because some candidate config depends on the entity identifier methods
        request_actions = {
            ENTITIES: self.request_entity_settings,
            CANDIDATES: self.request_candidate_settings
        }

        request_actions_to_execute = {}

        if hasattr(document_set.processing_information,
                   'step_for_restoring') and document_set.processing_information.step_for_restoring:
            current_index = document_set.processing_information.module_names.index(
                document_set.processing_information.step_for_restoring)

            for module in list(request_actions.keys()):
                try:
                    setting_index = document_set.processing_information.module_names.index(module)
                    if setting_index > current_index:
                        request_actions_to_execute.update({module: request_actions[module]})
                except ValueError:
                    continue

        else:
            request_actions_to_execute = {k:v for k,v in request_actions.items()
                                          if k in document_set.processing_information.module_names}

        default_config = True
        if config is not None:
            if self._run_config[CANDIDATES][CUSTOM_CAND_METHOD_NAME] is None:
                self._run_config[CANDIDATES][CUSTOM_CAND_METHOD_NAME] = copy.deepcopy(self._run_config[CANDIDATES][DEFAULT_CAND_METHOD])
            self.candidate_method = CUSTOM_CAND_METHOD_NAME

            self.read_config_params(config)
            self._run_config[CANDIDATES][self.candidate_method].read_annot(self.topic, self.cand_extraction_config.annot_index)
            self._run_config[ENTITIES][self.entity_method].get_default_params(self.def_params[self.entity_method])

            # self.read_config_params(config)
            if self.entity_identifier_config.param_source != DEFAULT:
                default_config = False
                method_dir = os.listdir(os.path.join(USER_CONFIG_SETTINGS, self.topic, ENTITIES))

                method_options = []
                for d in method_dir:
                    if self.entity_method == d.split("_")[0]:
                        method_options.append(d)
                # method_options.sort()
                try:
                    selected_config = method_options[self.entity_identifier_config.custom_files_id]
                    self._run_config[ENTITIES][self.entity_method].read_config(os.path.join(USER_CONFIG_SETTINGS,
                                                                    self.topic, ENTITIES, selected_config))
                except IndexError:
                    if len(method_options):
                        LOGGER.warning(MESSAGES["no_req_config_id"])
                        selected_config = method_options[-1]
                        self._run_config[ENTITIES][self.entity_method].read_config(os.path.join(USER_CONFIG_SETTINGS,
                                                            self.topic, ENTITIES, selected_config))
                    else:
                        LOGGER.warning(MESSAGES["no_config"].format(self.entity_method, self.topic) + " " +
                                       MESSAGES["no_param_config"])
                a=1
        else:
            # ask for config only for the left modules in the pipeline, i.e., don't ask for configuration for candidates
            # if a user restores the pipeline from entity identification module
            if len(request_actions_to_execute):
                default_config = self.request_interaction()

                if default_config:
                    self.save_default_params(request_actions_to_execute, default_config)
                else:
                    for interaction in list(request_actions_to_execute.values()):
                        interaction()

        for func_mod in sorted(list(request_actions_to_execute)):
            if func_mod == CANDIDATES:
                LOGGER.info(MESSAGES["cand_settings"] + "\n" +
                            "".join(["        " + k + ": " + str(v.name) + "\n" if type(v) != list
                                     else "        " + k + ": [" + ", ".join(t.name for t in v) + "]\n"
                                     for k, v in self.cand_extraction_config.__dict__.items() if
                                     issubclass(type(v), Enum)]) +
                            "".join(["        " + k + ": " + str(v) + "\n" for k, v in
                                     self.cand_extraction_config.__dict__.items() if
                                     not issubclass(type(v), Enum)]))

            if func_mod == ENTITIES:
                LOGGER.info(MESSAGES["ent_settings"] + "\n" +
                            "        method: " + self.entity_method + "\n" +
                            "        word_vectors: " + self.entity_identifier_config.word_vectors + "\n" +
                            "        param_source: " + self.entity_identifier_config.param_source + "\n")
                if self.entity_method == TCA_IMPROVED and default_config:
                    LOGGER.warning(MESSAGES["msma1_recom"])

            if func_mod == VISUALIZATION:
                LOGGER.info(MESSAGES["vis_settings"] + "\n" +
                            "        visualization: " + self.visualization_method + "\n")

    def read_config_params(self, config_params):
        # TODO introduce decent recursion
        if not len(config_params):
            return None
        orig_config = self.__dict__
        for r_key, r_values in orig_config.items():
            for p_key, p_values in config_params.items():
                if r_key != p_key:
                    continue
                if type(r_values) == dict:
                    for r_key_in, r_values_in in r_values.items():
                        for p_key_in, p_values_in in p_values.items():
                            if r_key_in != p_key_in:
                                continue
                            for r_key_in2, r_values_in2 in r_values_in.items():
                                for p_key_in2, p_values_in2 in p_values_in.items():
                                    if r_key_in2 != p_key_in2:
                                        continue
                                    for key, params in p_values_in2.items():
                                        # if key in r_values_in2.__dict__:
                                        setattr(orig_config[r_key][r_key_in][r_key_in2], key, params)
                else:
                    orig_config[r_key] = p_values
        for key, val in orig_config.items():
            setattr(self, key, val)

    @property
    def cand_extraction_config(self):
        return self._run_config[CANDIDATES][self.candidate_method]

    @property
    def entity_identifier_config(self):
        return self._run_config[ENTITIES][self.entity_method]

    @property
    def visualization_config(self):
        return self._run_config[VISUALIZATION][self.visualization_method]

    def request_interaction(self):
        reply = input(MESSAGES["user_interf"])
        return reply == "y"

    def request_entity_settings(self):
        # Choose a method for entity identification
        print(MESSAGES['methods'])
        for i, d in enumerate(list(self._run_config[ENTITIES].keys())):
            val = d if d != DEFAULT_ENTITY_METHOD else d + " (default)"
            print(str(i) + ": " + val)

        try:
            j = int(input(MESSAGES["sel_method"]))
            self.entity_method = list(self._run_config[ENTITIES].keys())[j]
        except (NameError, ValueError, IndexError):
            LOGGER.info(MESSAGES["no_entity_method"].format(self.entity_method.upper()))
            self.save_default_params([ENTITIES])
            return

        if self.topic not in os.listdir(USER_CONFIG_SETTINGS):
            LOGGER.info(MESSAGES["no_config"].format(self.entity_method.upper(), self.topic))
            LOGGER.info(MESSAGES["def_config"].format())
            self.save_default_params([ENTITIES])
            return

        # Choose a saved config file
        saved_dirs = os.listdir(os.path.join(USER_CONFIG_SETTINGS, self.topic, ENTITIES))
        if not any(self.entity_method in d.split("_")[0] for d in saved_dirs):
            LOGGER.info(MESSAGES["no_config"].format(self.entity_method.upper(), self.topic))
            LOGGER.info(MESSAGES["def_config"].format())
            self.save_default_params([ENTITIES])
            return

        method_dir = os.listdir(os.path.join(USER_CONFIG_SETTINGS, self.topic, ENTITIES))

        method_options = []
        for d in method_dir:
            if self.entity_method == d.split("_")[0]:
                method_options.append(d)
        method_options.sort()

        print((MESSAGES["available_folders"]))
        for i, d in enumerate(method_options):
            print(str(i) + ": " + d)
        try:
            j = int(input(MESSAGES["select_config"]))
            selected_config = method_options[j]
        except (NameError, ValueError, IndexError):
            LOGGER.info(MESSAGES["no_param_config"])  # TODO: Change back when Py 3.7 or higher
            return self.save_default_params([ENTITIES])

        self._run_config[ENTITIES][self.entity_method].get_default_params(os.path.join(ENTITIES, self.entity_method)
                                                                          if self.entity_method not in [XCOREF,
                                                                                                        XCOREF_BASE,
                                                                                                        TCA_IMPROVED,
                                                                                                        TCA_ORIG]
                                                                          else os.path.join(ENTITIES, SIEVE_BASED,
                                                                                            self.entity_method))
        self._run_config[ENTITIES][self.entity_method].read_config(os.path.join(USER_CONFIG_SETTINGS,
                                                                                self.topic, ENTITIES,
                                                                                selected_config))

    def request_candidate_settings(self):
        print(MESSAGES["cand_methods"])
        cand_params = {0: "real word setup",
                       1: "annotated candidates",
                       2: "default for " + self.entity_method.upper()
                       }

        custom_path = ""
        if self.topic in os.listdir(USER_CONFIG_SETTINGS):
            if CANDIDATES in os.path.join(USER_CONFIG_SETTINGS, self.topic):
                custom_path = os.path.join(os.path.join(USER_CONFIG_SETTINGS,
                                                        self.topic, CANDIDATES))
                for i, d in enumerate(os.listdir(custom_path)):
                    cand_params.update({i + len(cand_params): d})

        for i, d in cand_params.items():
            print(str(i) + ": " + d.replace("_", " "))

        try:
            selected_config = int(input(MESSAGES["sel_method"]))
        except (NameError, ValueError, IndexError):
            LOGGER.info(MESSAGES["no_param_config"])  # TODO: Change back when Py 3.7 or higher
            return self.save_default_params([CANDIDATES])

        if selected_config == 0:
            self.candidate_method = REALWORD_CAND_METHOD
        elif selected_config == 1:
            self.candidate_method = ANNOT_CAND_METHOD
        elif selected_config == 2:
            self.candidate_method = self.entity_method
        else:
            self.candidate_method = CUSTOM_CAND_METHOD_NAME
            self._run_config[CANDIDATES][CUSTOM_CAND_METHOD_NAME] = \
                ParamsCand(topic=self.topic).read_config(os.path.join(custom_path, cand_params[selected_config]))

    def save_default_params(self, modules, default_no_interaction=False):
        # if len(modules):
        #     logger.info(MESSAGES["def_config"])

        if ENTITIES in modules:
            self._run_config[ENTITIES][self.entity_method].get_default_params(os.path.join(ENTITIES, self.entity_method)
                                                                              if self.entity_method not in [XCOREF, XCOREF_BASE, TCA_IMPROVED, TCA_ORIG]
                                                                              else os.path.join(ENTITIES, SIEVE_BASED,
                                                                                                self.entity_method))
            if not default_no_interaction:
                reply = input(MESSAGES["save_def_config"].format(ENTITIES.upper()))
                if reply.lower() == "y":
                    # save entity config

                    if self.topic not in os.listdir(USER_CONFIG_SETTINGS):
                        os.mkdir(os.path.join(USER_CONFIG_SETTINGS, self.topic))

                    if ENTITIES not in os.listdir(os.path.join(USER_CONFIG_SETTINGS, self.topic)):
                        os.mkdir(os.path.join(USER_CONFIG_SETTINGS, self.topic, ENTITIES))

                    save_path = os.path.join(USER_CONFIG_SETTINGS, self.topic, ENTITIES,
                                             self.entity_method + "_" + NOW)
                    os.mkdir(save_path)
                    self._run_config[ENTITIES][self.entity_method].save_config(save_path)
                    LOGGER.info(MESSAGES["saved"].format(save_path))

        if CANDIDATES in modules:
            if not default_no_interaction:
                reply = input(MESSAGES["save_def_config"].format(CANDIDATES.upper()))
                if reply.lower() == "y":
                    # save candidates config

                    if self.topic not in os.listdir(USER_CONFIG_SETTINGS):
                        os.mkdir(os.path.join(USER_CONFIG_SETTINGS, self.topic))

                    if CANDIDATES not in os.listdir(os.path.join(USER_CONFIG_SETTINGS, self.topic)):
                        os.mkdir(os.path.join(USER_CONFIG_SETTINGS, self.topic, CANDIDATES))

                    save_file = os.path.join(USER_CONFIG_SETTINGS, self.topic, CANDIDATES,
                                             CANDIDATES + "_" + NOW + ".json")
                    self._run_config[CANDIDATES][self.candidate_method].save_config(save_file)
