from cdcr.structures.one_to_n_set import OneToNSet
from cdcr.structures.coref_mention import CorefMention
from typing import Iterable, Any


class CorefMentionSet(OneToNSet):
    CHILDRENS_PARENT_ATTRIBUTE = "mention_set"

    def __init__(self, items: Iterable = [], force: bool = False):
        """
        A set of mentions in a coreference.

        Args:
            items: Mentions to be contained in the set.
            force: Whether the mentions should be removed from another CorefMentionSet
                if already contained in another one.
        """
        try:
            super().__init__(items, force)
            self.coref = None
            """Coref: (Parent of this object.) Coref the mentions of this set belong to."""
        except TypeError:
            super().__init__(items.mention, force)
            self.coref = items


    def _test_and_parse(self, mention: Any, force: bool = False) -> Any:
        """
        Adding that a metions gets parsed to a CorefMention() if it is none already.

        See ``structures.one_to_n_set.OneToNSet._test_and_parse()``.
        """
        if type(mention) is not CorefMention:
            mention = CorefMention(mention)
        return super()._test_and_parse(mention, force)
