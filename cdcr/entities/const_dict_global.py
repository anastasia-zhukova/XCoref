# Entity identification menthods #
MSMA = "msma"
TCA_ORIG = "tca_orig"
TCA_IMPROVED = "tca_impr"
XCOREF = "xcoref"
XCOREF_HC = "xcoref_hc"
NLPA = "nlpa"
EECDCR = "eecdcr"
CORENLP = "corenlp"
CLUSTERING = "clustering"
LEMMA = "lemma"

# STEP CODES #
STEP = "step"
STEPS_INIT = [i for i in range(7)]
STEP0 = STEP + str(STEPS_INIT[0])
STEP1 = STEP + str(STEPS_INIT[1])
STEP2 = STEP + str(STEPS_INIT[2])
STEP3 = STEP + str(STEPS_INIT[3])
STEP4 = STEP + str(STEPS_INIT[4])
STEP5 = STEP + str(STEPS_INIT[5])
STEP6 = STEP + str(STEPS_INIT[6])

# STEP NAMES #
INIT_STEP = "init"

MSMA1_0 = "0_representative_words"
MSMA1_1 = "1_core_similarity"
MSMA1_2 = "2_labeling_similarity"
MSMA1_3 = "3_compound_similarity"
MSMA1_4 = "4_freq_wordset_similarity"
MSMA1_5 = "5_freq_phrase_similarity"

MSMA2_0 = "0_wikipage_match"
MSMA2_1 = "1_core_NE_mentions_match"
MSMA2_2 = "2_core_mentions_match"
MSMA2_3 = "3_person-non-ne_phrasing_similarity"
MSMA2_3_SUB_0 = "_a_common_phrase"
MSMA2_3_SUB_1 = "_b_common_head"
MSMA2_3_SUB_2 = "_0_core"
MSMA2_3_SUB_3 = "_1_non_core"
MSMA2_3_SUB_4 = "_2_body"
MSMA2_3_SUB_5 = "_3_border"
MSMA2_3_SUB_6 = "_c_match_adj"
MSMA2_4 = "4_prominent_wording_similarity"
MSMA2_5 = "5_misc_type_similarity"
MSMA2_6 = "6_meronym_similarity"
MSMA2_7 = "7_context_similarity"
MSMA3_2 = "2_core_mentions_match"
MSMA3_3 = "3_person-non-ne_phrasing_similarity"
MSMA3_4 = "4_misc_type_similarity"

# ENTITY PROPERTY #
ORIGINAL_ENTITY = "original"
NAME = "name"
SIZE = "size"
TYPE = "type"
PHRASING_COMPLEXITY = "phrasing_complexity"
PHRASES = "phrases"
MERGE_HISTORY = "merging_history"
MERGED_ENTITIES = "merged_entities"
SIM_SCORE = "sim_score"
REPRESENTATIVE = "representative"

# ENTITY TYPES #
PERSON_NE_TYPE = "person-ne"
PERSON_NES_TYPE = "person-nes"
PERSON_TYPE = "person"
PERSON_NN_TYPE = "person-nn"
PERSON_NNS_TYPE = "person-nns"
GROUP_NE_TYPE = "group-ne"
GROUP_TYPE = "group"
COUNTRY_NE_TYPE = "country-ne"
COUNTRY_TYPE = "country"
MISC_TYPE = "misc"

# ENTITY TYPES #
NE_TYPE = "ne"
NES_TYPE = "nes"
NN_TYPE = "nn"
NNS_TYPE = "nns"
NON_NE_TYPE = "non-ne"

# DEPENDENCY TYPES
AMOD = "amod"
NMOD = "nmod"
COMPOUND = "compound"
NUMMOD = "nummod"
NMOD_POSS = "nmod:poss"
APPOS = "appos"
ACL = "acl"
ACL_RELCL = "acl:relcl"
PUNCT = "punct"
DOBJ = "dobj"

# WORDNET TYPES
PERSON_WN = "noun.person"
GROUP_WN = "noun.group"
LOCATION_WN = "noun.location"
QUANTITY_WN = "noun.quantity"
ACT_WN = "noun.act"

# NER_TYPES
NON_NER = "O"
PERSON_NER = "PERSON"
ORGANIZATION_NER = "ORGANIZATION"
COUNTRY_NER = "COUNTRY"
DATE_NER = "DATE"
TIME_NER = "TIME"
DURATION_NER = "DURATION"
TITLE_NER = "TITLE"
NATIONALITY_NER = "NATIONALITY"
STATE_NER = "STATE_OR_PROVINCE"
CITY_NER = "CITY"
MISC_NER = "MISC"
IDEOLOGY_NER = "IDEOLOGY"
LOCATION_NER = "LOCATION"

# Dependent trees
DEP = "dep"
DEPENDENT_GLOSS = "dependentGloss"
DEPENDENT = "dependent"
GOVERNOR_GLOSS = "governorGloss"
GOVERNOR = "governor"

# POS tags and coref types
PRONOMINAL = "PRONOMINAL"
DT = "DT"
NN = "NN"
POS = "POS"
JJ = "JJ"

# PARAMS
CUSTOM = "custom"
DEFAULT = "default"
