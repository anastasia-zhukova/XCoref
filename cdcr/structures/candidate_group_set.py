from cdcr.candidates.cand_enums import *

from cdcr.structures.one_to_n_set import OneToNSet
from typing import Iterable, Any
from cdcr.structures.candidate import Candidate
# from newsalyze.structures.candidate_set import CandidateSet


class CandidateGroupSet(OneToNSet):
    CHILDRENS_PARENT_ATTRIBUTE = "candidate_group_set"

    def __init__(self, group_name: str, items: Iterable = [], force: bool = True):
        """
        A set of related candidates, connected with coref dependency.

        Args:
            group_name: Name of a coreference chain representative.
            items: Candidates to be contained in the set.
            force: Whether the candidates should be removed from another CandidateSet if already contained in another one.
        """
        self.group_name = group_name

        super().__init__(items, force)

    def _test_and_parse(self, candidate: Any, force: bool = True) -> Candidate:
        """
        See ``structures.one_to_n_set.OneToNSet._test_and_parse()``.
        """
        if type(candidate) is not Candidate:
            raise TypeError("The class passed into " + str(self.__class__) + " is not " + str(Candidate.__class__) +
                            " but " + candidate.__class__)
        return super()._test_and_parse(candidate, force)
