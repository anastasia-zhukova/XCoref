from cdcr.structures.wrapper import Wrapper

from stanfordnlp.protobuf import NERMention as ProtobufNERMention


class NERMention(Wrapper):

    WRAPPED_ATTRIBUTE = "mention"

    def __init__(self, mention: ProtobufNERMention):
        """
        A mention which references another mention.

        Args:
            mention: Protobufs coref-mention this object should represent.
        """
        self.mention = mention
        """ProtobufNERMention: Mention this object represents and falls back to."""

        self.mention_set = None
        """NERMentionSet: (Parent of this object.) The set of NERs this mention is contained in."""

    @property
    def tokens(self):
        sentence_tokens = self.mention_set.sentence.tokens
        start = self.tokenStartInSentenceInclusive - sentence_tokens[0].tokenBeginIndex
        end = self.tokenEndInSentenceExclusive - sentence_tokens[0].tokenBeginIndex
        return sentence_tokens[start:end]

    @property
    def text(self):
        tokens = self.tokens
        start = tokens[0].beginChar
        end = tokens[-1].endChar
        text = self.mention_set.sentence.sentence_set.document.fulltext
        return text[start:end]
