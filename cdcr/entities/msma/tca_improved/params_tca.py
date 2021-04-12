from cdcr.entities.msma.params_msma import *
from cdcr.entities.const_dict_global import *
from cdcr.config import *


VERY_SIMILAR_ADD_VALUE_NAME = "very_similar_add_value"


class ParamsTCA(MSMAParams):
    """
    A class with configuration parameters for MSMA 1.0
    """

    word_vectors = WORD2VEC_WE

    def add_params(self, params):
        params["steps_to_execute"] = [STEP + str(i) for i in range(len(STEPS_INIT))]

        # entity property
        params["entity_property"] = Params({
            "damping": 0.7,
            "max_iter": 100,
            "min_support": 4,
            "very_freq_item_threshold": 0.9,
            "representative_word_num": 6
        })

        # round2: merge using labeling
        params[STEP2] = Params({
            "merge_labeling_proportion_threshold": 0.3,
            "merge_labeling_min_dim_number": 2,
            VERY_SIMILAR_ADD_VALUE_NAME: 0.2
        })

        # round3: merge using compound labeling
        params[STEP3] = Params({
            "merge_compound_proportion_threshold": 0.5
        })

        # round4: merge using frequent wordsets
        params[STEP4] = Params({
            "merge_freq_phrases_proportion": 0.3,
            VERY_SIMILAR_ADD_VALUE_NAME: 0.2
        })

        # round5: merge using common strings
        params[STEP5] = Params({
            "merge_string_proportion_threshold": 0.7,
            VERY_SIMILAR_ADD_VALUE_NAME: 0.2
        })
        return Params(params)

    def get_wordvectors_type(self) -> str:
        return WORD2VEC_WE
