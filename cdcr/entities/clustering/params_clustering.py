from cdcr.entities.params_entities import *
from cdcr.config import *


class ParamsClustering(ParamsEntities):
    """
    Parameters for a clustering-based entity identifier.
    """

    def get_default_params(self, def_folder_path=None):
        params = {"min_sim": 0.6}
        self.params.update(params)
        self.word_vectors = WORD2VEC_WE if self.word_vectors == "not_specified" else self.word_vectors
        return self
