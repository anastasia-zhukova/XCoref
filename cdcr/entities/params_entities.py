from cdcr.structures.params import *
from cdcr.entities.const_dict_global import *


class ParamsEntities(ParamsHelperClass):
    """
    Configuration parameters for entity identifiers.

    """
    def __init__(self):
        params = dict()

        # preprocessing parameters
        params["preprocessing"] = Params({
            "check_coref_groups": True,
            "exclude_pronouns": True,
            "exclude_time": True,
            "exclude_general_nouns": True,
            "exclude_dt": True,
            "exclude_single_adj": True,
            "infreq_entity_type_threshold": 0.25
        })

        # evaluation parameters
        params["evaluation"] = Params({"evaluation_mode": True})
        self.params = Params(params)
        self.param_source = DEFAULT
        self.custom_files_id = None
        self.word_vectors = "not_specified"
        self.load_preproc = False

    def read_config(self, config_path):
        with open(os.path.join(config_path, self.__class__.__name__ + ".json"), "r") as file:
            json_values = json.load(file)
            out = {}
            for k,v in json_values.items():
                if type(v) == dict:
                    out.update({k, Params(v)})
                else:
                    out.update({k:v})
            self.params = Params(out)
        self.param_source = CUSTOM

    def save_config(self, config_path):
        out = {}
        for k, v in self.params.__dict__.items():
            if issubclass(v.__class__, Params):
                out.update({k: v.__dict__})
            else:
                out.update({k:v})

        with open(os.path.join(config_path, self.__class__.__name__ + ".json"), "w") as file:
            json.dump(out, file)
