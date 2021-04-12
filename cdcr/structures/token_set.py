from cdcr.structures.one_to_n_set import OneToNSet
from cdcr.structures.token import Token
from typing import Iterable, Any


class TokenSet(OneToNSet):

    CHILDRENS_PARENT_ATTRIBUTE = "token_set"

    def __init__(self, items: Iterable = [], force: bool = False):
        """
        A set of tokens in a sentence.

        Args:
            items: Tokens contained in the set.
            force: Whether the tokens should be removed from another TokenSet if already contained in another one.
        """
        try:
            super().__init__(items, force)
            self.sentence = None
            """Document: (Parent of this object.) Document the corefs in this set belong to."""
        except TypeError:
            super().__init__(items.token, force)
            self.sentence = items

    def _test_and_parse(self, token: Any, force: bool = False) -> Token:
        """
        Adding that a sentence gets parsed to a Sentence() if it is not already.

        See ``structures.one_to_n_set.OneToNSet._test_and_parse()``.
        """
        if type(token) is not Token:
            token = Token(token)
        return super()._test_and_parse(token, force)
