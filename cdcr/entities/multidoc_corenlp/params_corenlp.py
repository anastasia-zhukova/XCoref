from cdcr.entities.params_entities import *


class ParamsCoreNLP(ParamsEntities):
    """
    A class with parametes required for execution of CoreNLP entity identifer.
    """

    def get_default_params(self, def_folder_path=None):
        return self
