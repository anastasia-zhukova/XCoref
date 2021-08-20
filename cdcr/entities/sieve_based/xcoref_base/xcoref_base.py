from cdcr.entities.sieve_based.sieve_based import SieveBasedApproach
from cdcr.entities.sieve_based.xcoref.steps.step2_0 import XCorefStep0
from cdcr.entities.sieve_based.xcoref.steps.step2_1 import XCorefStep1
from cdcr.entities.sieve_based.xcoref.steps.step2_2 import XCorefStep2
from cdcr.entities.sieve_based.xcoref_base.steps.step3_3 import XCoref_Base_Step3
from cdcr.entities.sieve_based.xcoref_base.steps.step3_4 import XCoref_Base_Step4
from cdcr.entities.const_dict_global import *
from cdcr.entities.sieve_based.xcoref.entity_xcoref import EntityXCoref


class XCoref_Base(SieveBasedApproach):
    """
    A variant of XCoref with simpler menthods for the last two steps.
    """

    def __init__(self, docs):
        super().__init__(docs, EntityXCoref)

        self.steps = {STEP0: XCorefStep0,
                      STEP1: XCorefStep1,
                      STEP2: XCorefStep2,
                      STEP3: XCoref_Base_Step3,
                      STEP4: XCoref_Base_Step4}
