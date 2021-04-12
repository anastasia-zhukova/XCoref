from cdcr.entities.msma.msma import MultiStepMergingApproach
from cdcr.entities.msma.tca_improved.entity_tca import EntityTCA
from cdcr.entities.const_dict_global import *
from cdcr.entities.msma.tca_improved.steps.step1_0 import TCAStep0
from cdcr.entities.msma.tca_improved.steps.step1_1 import TCAStep1
from cdcr.entities.msma.tca_improved.steps.step1_2 import TCAStep2
from cdcr.entities.msma.tca_improved.steps.step1_3 import TCAStep3
from cdcr.entities.msma.tca_improved.steps.step1_4 import TCAStep4
from cdcr.entities.msma.tca_improved.steps.step1_5 import TCAStep5
from cdcr.entities.entity_preprocessor import EntityPreprocessor


class TargetConceptAnalysisOriginal(MultiStepMergingApproach):
    """
    Target COncept Analysis (TCA) original implementation.
    """

    def __init__(self, docs):
        super().__init__(docs, EntityTCA, EntityPreprocessor)

        self.steps = {STEP0: TCAStep0,
                      STEP1: TCAStep1,
                      STEP2: TCAStep2,
                      STEP3: TCAStep3,
                      STEP4: TCAStep4,
                      STEP5: TCAStep5}
