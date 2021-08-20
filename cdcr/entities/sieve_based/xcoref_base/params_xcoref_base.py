from cdcr.entities.sieve_based.params_sieves import *
from cdcr.entities.const_dict_global import *
from cdcr.config import *


EPS = "eps"


class ParamsXCorefBase(SievesParams):
    """
    A class with configuration parameters for SIEVE_BASED 2.0
    """

    word_vectors = GLOVE_WE

    def add_params(self, params: Dict):
        params["steps_to_execute"] = [STEP + str(i) for i in range(len(STEPS_INIT))]

        # STEP2: merge using cosine similarity to NE mentions
        params[STEP2] = Params({"mention_similarity_threshold": 0.6,
                                EPS: 0.01})

        # STEP3: merge step for person-non-NE-related entities
        params[STEP3] = Params({
            "min_sim": 0.3,
            "weight_words": False})

        # STEP4: merge step for misc type
        params[STEP4] = Params({
            "min_sim_same_type": 0.4,
            "min_sim_diff_type": 0.7
        })

        return Params(params)

    def get_wordvectors_type(self) -> str:
        return GLOVE_WE
