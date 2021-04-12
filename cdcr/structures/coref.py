from graphene import ObjectType, List as GQLList

from cdcr.structures.coref_mention_set import CorefMentionSet
from cdcr.structures.coref_mention import CorefMention
from stanfordnlp.protobuf import CorefChain as ProtobufCorefChain

from cdcr.structures.graphene_schema import SCHEMA
from cdcr.structures.wrapper import Wrapper
from cdcr.util.graphene import FieldProperty


class Coref(Wrapper):
    WRAPPED_ATTRIBUTE = "coref"

    def __init__(self, coref: ProtobufCorefChain):
        """
        Coreference of a document.

        Args:
            coref: Protobufs coref this object should represent.
        """

        self.mentions = None
        """CorefMentionSet: List of mentions this coref contains."""

        self.coref_set = None
        """CorefSet: (Parent of this object.) The set of coreferences this coref is contained in."""

        self.coref = coref
        """ProtobufCorefChain: Coreference this object represents and falls back to."""

    @property
    def representative(self) -> CorefMention:
        """Representative mention of this coref."""
        return self.mentions[self.coref.representative]

    @property
    def id(self) -> int:
        """ID of the coref."""
        return self.chainID

    @property
    def coref(self) -> ProtobufCorefChain:
        """Protobuf coref chained represented by this object."""
        return self.__coref

    @coref.setter
    def coref(self, coref: ProtobufCorefChain):
        self.__coref = coref
        if coref:
            self.mentions = CorefMentionSet(self)
