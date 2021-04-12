from cdcr.candidates.cand_enums import *

from cdcr.structures.one_to_n_set import OneToNSet
from typing import Iterable, Any
from cdcr.structures.candidate import Candidate
from cdcr.structures.candidate_group_set import CandidateGroupSet


class CandidateSet(OneToNSet):
    CHILDRENS_PARENT_ATTRIBUTE = "candidate_set"

    def __init__(self, origin_type: OriginType = OriginType.EXTRACTED,
                 coref_strategy: CorefStrategy = CorefStrategy.ONE_DOC, items: Iterable = [], force: bool = True):
        """
        A set of extracted groups of candidates.

        Args:
            items: CandidateGroupSets to be contained in the set.
            force: Whether the candidates should be removed from another CandidateGroupSet if already contained in another one.
        """
        super().__init__(items, force)

        self.origin_type = origin_type
        self.coref_strategy = coref_strategy

    def _test_and_parse(self, candidate_group: Any, force: bool = True) -> CandidateGroupSet:
        """
        See ``structures.one_to_n_set.OneToNSet._test_and_parse()``.
        """
        if type(candidate_group) is not CandidateGroupSet:
            # candidate_group = CandidateGroupSet(candidate_group)
            raise TypeError("The class passed into " + str(self.__class__) + " is not " +
                            str(CandidateGroupSet.__class__) + " but " + candidate_group.__class__)
        return super()._test_and_parse(candidate_group, force)
