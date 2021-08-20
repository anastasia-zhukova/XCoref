from cdcr.entities.sieve_based.params_sieves import *
from cdcr.entities.const_dict_global import *
from cdcr.config import *


EPS = "eps"


class ParamsXCoref(SievesParams):
    """
    A class with configuration parameters for SIEVE_BASED 2.0 + 1.0
    """

    word_vectors = GLOVE_WE

    def add_params(self, params: Dict):
        params["steps_to_execute"] = [STEP + str(i) for i in range(len(STEPS_INIT))]

        # STEP2: merge using cosine similarity to NE mentions
        params[STEP2] = Params({"mention_similarity_threshold": 0.6,
                                EPS: 0.01})

        # STEP3: merge step for person-non-NE-related entities
        params[STEP3] = Params({
            "max_repr_words": 4,
            "ne_word_weight": 1.7,
            "min_sim": 0.4,
            "min_core_sim": 0.4,
            "overlap_ratio_max": 0.7,
            "overlap_ratio_min": 0.5,
            "merge_phrases_weight": 0.5,
            "aliens_phrases_weight": 0.4,
            "min_cluster_sim": 0.5,
            "min_border_match": 2,
            EPS: 0.01,
            "min_feature_ratio": 0.6,
            "group_ne_sim": 0.7
        })

        # STEP4: merge step for misc type
        params[STEP4] = Params({
            "min_sim": 0.4,
            "weight_words": True
        })

        return Params(params)

    def get_wordvectors_type(self) -> str:
        return GLOVE_WE
