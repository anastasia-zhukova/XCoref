from cdcr.structures.one_to_n_set import OneToNSet
from cdcr.structures.ner_mention import NERMention
from typing import Iterable, Any
from cdcr.structures.embedding import PassListEmbedding


class NERMentionSet(OneToNSet):
    CHILDRENS_PARENT_ATTRIBUTE = "mention_set"

    def __init__(self, items: Iterable = [], force: bool = False):
        """
        A set of NEs in a document.

        Args:
            items: NEs to be contained in the set.
            force: Whether the NEs should be removed from another ner_set if already contained in another one.
        """
        try:
            super().__init__(items, force)
            self.sentence = None
            """Document: (Parent of this object.) Document the NEs in this set belong to."""
        except TypeError:
            super().__init__(items.mentions, force)
            self.sentence = items

    def _test_and_parse(self, ner_mention: Any, force: bool = False) -> NERMention:
        """
        Adding that a corefs gets parsed to a Coref() if it is none already.

        See ``structures.one_to_n_set.OneToNSet._test_and_parse()``.
        """
        if type(ner_mention) is not NERMention:
            ner_mention = NERMention(ner_mention)
        return super()._test_and_parse(ner_mention, force)
