from cdcr.structures.wrapper import Wrapper
from stanfordnlp.protobuf import Token as ProtobufToken


class Token(Wrapper):
    WRAPPED_ATTRIBUTE = "token"

    def __init__(self, token: ProtobufToken):
        """
        A token contained in a sentence

        Args:
            token: Protobufs Token this object should represent.
        """
        self.token = token
        """ProtobufToken: Protobuf Token this object represents."""

        self.token_set = None
        """TokenSet: (Parent of this object.) List of tokens (which yield a sentence) this token is contained in."""

    def __repr__(self):
        return self.word

    @property
    def index(self):
        """Index of the token in the sentence."""
        return self.token_set.index(self)

    @property
    def sentence_begin_char(self):
        """Beginning character position of the token in the sentence."""
        sentence = self.token_set.sentence
        return self.beginChar-sentence.begin_char

    @property
    def sentence_end_char(self):
        """Ending character position of the token in the sentence."""
        sentence = self.token_set.sentence
        return self.endChar-sentence.begin_char
