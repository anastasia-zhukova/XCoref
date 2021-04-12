from cdcr.entities.msma.params_msma import *
from cdcr.entities.const_dict_global import *
from cdcr.config import *


EPS = "eps"


class ParamsMSMA3(MSMAParams):
    """
    A class with configuration parameters for MSMA 2.0
    """

    word_vectors = ELMO_WE

    def add_params(self, params: Dict):
        params["steps_to_execute"] = [STEP + str(i) for i in range(len(STEPS_INIT))]

        # STEP2: merge using cosine similarity to NE mentions
        # params[STEP2] = Params({"mention_similarity_threshold": 0.6})
        params[STEP2] = Params({"mention_similarity_threshold": 0.6,
                                EPS: 0.01})

        # STEP3: merge step for person-non-NE-related entities
        params[STEP3] = Params({
            "min_sim": 0.4})

        # STEP4: merge step for misc type
        params[STEP4] = Params({
            "min_sim": 0.4})

        return Params(params)

    def get_wordvectors_type(self) -> str:
        return ELMO_WE
