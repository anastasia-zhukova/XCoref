from cdcr.entities.msma.params_msma import *
from cdcr.entities.const_dict_global import *
from cdcr.config import *


EPS = "eps"


class ParamsMSMA2_1(MSMAParams):
    """
    A class with configuration parameters for MSMA 2.0 + 1.0
    """

    word_vectors = WORD2VEC_WE

    def add_params(self, params: Dict):
        params["steps_to_execute"] = [STEP + str(i) for i in range(len(STEPS_INIT))]

        # STEP2: merge using cosine similarity to NE mentions
        params[STEP2] = Params({"mention_similarity_threshold": 0.6,
                                EPS: 0.01})

        # STEP3: merge step for person-non-NE-related entities
        params[STEP3] = Params({
            "max_repr_words": 4,
            "ne_word_weight": 1.7,
            "min_sim": 0.4 if self.word_vectors != ELMO_WE else 0.5,
            "min_core_sim": 0.4 if self.word_vectors != ELMO_WE else 0.5,
            "overlap_ratio_max": 0.7,
            "overlap_ratio_min": 0.5,
            "merge_phrases_weight": 0.5,
            "aliens_phrases_weight": 0.4,
            "min_cluster_sim": 0.5,
            "min_border_match": 2,
            EPS: 0.01,
            "min_feature_ratio": 0.6
        })

        # STEP4: merge step for misc type
        params[STEP4] = Params({
            "min_sim_same_type": 0.4,
            "min_sim_diff_type": 0.7
        })

        return Params(params)

    def get_wordvectors_type(self) -> str:
        return WORD2VEC_WE
