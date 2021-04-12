from cdcr.entities.msma.msma import MultiStepMergingApproach
from cdcr.entities.msma.xcoref.steps.step2_0 import XCorefStep0
from cdcr.entities.msma.xcoref.steps.step2_1 import XCorefStep1
from cdcr.entities.msma.xcoref.steps.step2_2 import XCorefStep2
from cdcr.entities.msma.xcoref_hc.steps.step3_3 import XCoref_HC_Step3
from cdcr.entities.msma.xcoref_hc.steps.step3_4 import XCoref_HC_Step4
from cdcr.entities.const_dict_global import *
from cdcr.entities.msma.xcoref.entity_xcoref import EntityXCoref


class XCoref_HierarClust(MultiStepMergingApproach):
    """
    A combination of XCoref and Hierarchical clustering that replaces two steps of XCoref
    """

    def __init__(self, docs):
        super().__init__(docs, EntityXCoref)

        self.steps = {STEP0: XCorefStep0,
                      STEP1: XCorefStep1,
                      STEP2: XCorefStep2,
                      STEP3: XCoref_HC_Step3,
                      STEP4: XCoref_HC_Step4}
