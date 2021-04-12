from cdcr.structures.one_to_n_set import OneToNSet
from cdcr.structures.coref import Coref
from typing import Iterable, Any, Union
from cdcr.structures.embedding import PassListEmbedding


class CorefSet(OneToNSet):
    CHILDRENS_PARENT_ATTRIBUTE = "coref_set"

    def __init__(self, items: Union[Iterable, 'Document'] = [], force: bool = False):
        """
        A set of coreferences in a document.

        Args:
            items: Coreferences to be contained in the set.
            force: Whether the coreferences should be removed from another CorefSet if already contained in another one.
        """
        try:
            super().__init__(items, force)
            self.document = None
            """Document: (Parent of this object.) Document the corefs in this set belong to."""
        except TypeError:
            super().__init__(items.corefChain, force)
            self.document = items

    def _test_and_parse(self, coref: Any, force: bool = False) -> Coref:
        """
        Adding that a corefs gets parsed to a Coref() if it is none already.

        See ``structures.one_to_n_set.OneToNSet._test_and_parse()``.
        """
        if type(coref) is not Coref:
            coref = Coref(coref)
        return super()._test_and_parse(coref, force)
