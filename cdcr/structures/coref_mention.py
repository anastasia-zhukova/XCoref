from graphene import ObjectType, Field, Boolean, String
from stanfordnlp.protobuf import Mention as ProtobufCorefMention, Token as ProtobufToken

from cdcr.structures.graphene_schema import SCHEMA
from cdcr.structures.token import Token
from cdcr.structures.wrapper import Wrapper
from cdcr.structures.sentence import Sentence
from typing import List

from cdcr.util.graphene import FieldProperty


class CorefMention(Wrapper):

    WRAPPED_ATTRIBUTE = "mention"

    def __init__(self, mention: ProtobufCorefMention):
        """
        A mention which references another mention.

        Args:
            mention: Protobufs coref-mention this object should represent.
        """
        self.mention = mention
        """ProtobufCorefMention: Mention this object represents and falls back to."""

        self.mention_set = None
        """CorefMentionSet: (Parent of this object.) The set of mentions this mention is contained in."""

    @property
    def is_representative(self) -> bool:
        """Whether this is the representative mention in the coref."""
        return self.mention_set.coref.representative is self

    @property
    def is_representative_mention(self) -> bool:
        return self.is_representative

    @property
    def sentence(self) -> Sentence:
        """Sentence this mention is contained in."""
        document = self.mention_set.coref.coref_set.document
        return document.sentences[self.sentenceIndex]

    @property
    def tokens(self) -> List[Token]:
        """Tokens contained in this mention."""
        return self.sentence.tokens[self.beginIndex:self.endIndex]

    @property
    def text(self) -> str:
        """Text of this mention."""
        text = self.mention_set.coref.coref_set.document.fulltext
        tokens = self.tokens
        return text[tokens[0].beginChar:tokens[-1].endChar]

    @property
    def head_token(self) -> Token:
        """The head token of this mention."""
        return self.sentence.tokens[self.headIndex]

    def __repr__(self):
        return self.text
