from cdcr.entities.msma.msma import MultiStepMergingApproach
from cdcr.entities.msma.xcoref.steps.step2_0 import XCorefStep0
from cdcr.entities.msma.xcoref.steps.step2_1 import XCorefStep1
from cdcr.entities.msma.xcoref.steps.step2_2 import XCorefStep2
from cdcr.entities.msma.xcoref.steps.step2_3 import XCorefStep3
from cdcr.entities.msma.xcoref.steps.step2_4 import XCorefStep4
from cdcr.entities.const_dict_global import *
from cdcr.entities.msma.xcoref.entity_xcoref import EntityXCoref


class XCoref(MultiStepMergingApproach):
    """
    XCoref: New approach with a combination of step 1 from TCA
    """

    def __init__(self, docs):
        super().__init__(docs, EntityXCoref)

        self.steps = {STEP0: XCorefStep0,
                      STEP1: XCorefStep1,
                      STEP2: XCorefStep2,
                      STEP3: XCorefStep3,
                      STEP4: XCorefStep4}
