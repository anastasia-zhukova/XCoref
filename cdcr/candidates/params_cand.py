from cdcr.candidates.cand_enums import *
from cdcr.config import *
from cdcr.structures.params import *


TYPE_MAPPING = [OriginType, CorefStrategy, ExtentedPhrases, CandidateType, ChangeHead]
ANNOTATION = "annotation"


class ParamsCand(ParamsHelperClass):

    def __init__(self, topic):
        self.origin_type = None
        self.coref_extraction_strategy = None
        self.phrase_extension = None
        self.add_phrases = None
        self.change_head = None
        self.max_phrase_len = 25
        self.annot_path = None
        self.annot_index = -1
        self.drop_duplicates = True
        self.ignore_lists = True
        self.max_doc_num = 5
        self.read_annot(topic, self.annot_index)

    def read_annot(self, topic, annot_index=-1):
        if ANNOTATION in os.listdir(os.path.join(ORIGINAL_DATA_PATH, topic)):
            annot_path = os.path.join(ORIGINAL_DATA_PATH, topic, ANNOTATION)
            folder_list = os.listdir(annot_path)
            try:
                self.annot_path = os.path.join(annot_path, folder_list[annot_index])
                self.annot_index = annot_index
            except IndexError:
                if len(folder_list):
                    LOGGER.warning("Requested annotation folder not found. Last annotation will be used.")
                    self.annot_path = os.path.join(annot_path, folder_list[-1])
                    self.annot_index = -1
                else:
                    self.annot_path = None
                    self.annot_index = None
        else:
            self.annot_index = None

    def get_corenlp_params(self):
        self.origin_type = OriginType.EXTRACTED_ANNOTATED
        self.coref_extraction_strategy = CorefStrategy.MULTI_DOC
        self.phrase_extension = ExtentedPhrases.ASIS
        self.add_phrases = [CandidateType.NP, CandidateType.VP]
        self.change_head = ChangeHead.ORIG
        self.ignore_lists = False
        self.max_doc_num = 8
        return self

    def get_default_params(self, def_folder_path=None):
        self.origin_type = OriginType.EXTRACTED_ANNOTATED
        self.coref_extraction_strategy = CorefStrategy.MULTI_DOC
        self.phrase_extension = ExtentedPhrases.EXTENDED
        self.add_phrases = [CandidateType.NP, CandidateType.VP]
        self.change_head = ChangeHead.NON_NUMBER
        return self

    def get_realword_setup(self):
        self.origin_type = OriginType.EXTRACTED
        self.coref_extraction_strategy = CorefStrategy.ONE_DOC
        self.phrase_extension = ExtentedPhrases.EXTENDED
        self.add_phrases = [CandidateType.NP, CandidateType.VP]
        self.change_head = ChangeHead.NON_NUMBER
        return self

    def get_annot_setup(self):
        self.origin_type = OriginType.ANNOTATED
        self.coref_extraction_strategy = CorefStrategy.ONE_DOC
        self.phrase_extension = ExtentedPhrases.ASIS
        self.add_phrases = [CandidateType.NP]
        self.change_head = ChangeHead.ORIG
        return self

    def get_barhom_params(self):
        self.origin_type = OriginType.ANNOTATED
        self.coref_extraction_strategy = CorefStrategy.ONE_DOC
        self.phrase_extension = ExtentedPhrases.ASIS
        self.add_phrases = [CandidateType.NP]
        self.change_head = ChangeHead.ORIG
        return self

    def get_nlpa_params(self):
        self.origin_type = OriginType.ANNOTATED
        self.coref_extraction_strategy = CorefStrategy.NO_COREF
        self.phrase_extension = ExtentedPhrases.ASIS
        self.add_phrases = []
        self.change_head = ChangeHead.ORIG
        return self

    def get_lemma_params(self):
        self.origin_type = OriginType.ANNOTATED
        self.coref_extraction_strategy = CorefStrategy.NO_COREF
        self.phrase_extension = ExtentedPhrases.EXTENDED
        self.add_phrases = []
        self.change_head = ChangeHead.ORIG
        return self

    def read_config(self, path):
        with open(path) as file:
            custom_config = json.load(file)
        for i, (key, vals) in enumerate(custom_config.items()):
            if type(vals) == list:
                conv_vals = []
                for val in vals:
                    conv_vals.append(TYPE_MAPPING[i].from_string(val))
                setattr(self, key, conv_vals)
            else:
                setattr(self, key, TYPE_MAPPING[i].from_string(vals))
        return self

    def save_config(self, path):
        modif_dict = {}
        for k, v in self.__dict__.items():
            modif_dict[k] = v.name
        with open(path, "w") as file:
            json.dump(modif_dict, file)
