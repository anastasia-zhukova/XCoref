from cdcr.structures.one_to_n_set import OneToNSet
from cdcr.structures.sentence import Sentence
from typing import Iterable, Any


class SentenceSet(OneToNSet):

    CHILDRENS_PARENT_ATTRIBUTE = "sentence_set"

    def __init__(self, items: Iterable = [], force: bool = False):
        """
        A set of sentences in a document.

        Args:
            items: Sentences to be contained in the set.
            force: Whether the sentences should be removed from another SentenceSet if already contained in another one.
        """
        try:
            super().__init__(items, force)
            self.document = None
            """Document: (Parent of this object.) Document the corefs in this set belong to."""
        except TypeError:
            super().__init__(items.sentence, force)
            self.document = items

    def _test_and_parse(self, sentence: Any, force: bool = False) -> Sentence:
        """
        Adding that a sentence gets parsed to a Sentence() if it is not already.

        See ``structures.one_to_n_set.OneToNSet._test_and_parse()``.
        """
        if type(sentence) is not Sentence:
            sentence = Sentence(sentence)
        return super()._test_and_parse(sentence, force)
