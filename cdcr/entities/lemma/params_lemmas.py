from cdcr.entities.params_entities import *
from cdcr.config import *


class ParamsLemmas(ParamsEntities):
    """
    Parameters for a clustering-based entity identifier.
    """

    def get_default_params(self, def_folder_path=None):
        # params = {"min_sim": 0.6}
        # self.params.update(params)
        # self.word_vectors = ELMO_WE
        return self
