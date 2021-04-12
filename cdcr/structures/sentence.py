from stanfordnlp.protobuf import Sentence as ProtobufSentence
from cdcr.structures.wrapper import Wrapper
from cdcr.structures.tree import ParseTree
from cdcr.structures.dependencies import DependencyGraph
from stanfordnlp.protobuf import Sentence as ProtobufSentence
from cdcr.structures.token_set import TokenSet
from cdcr.structures.ner_mention_set import NERMentionSet


class Sentence(Wrapper):
    WRAPPED_ATTRIBUTE = "_sentence"

    def __init__(self, sentence: ProtobufSentence):
        """
        A sentence which is contained in a sentence set.

        Args:
            sentence: Protobufs Sentence this object should represent.
        """
        self.sentence = sentence
        """ProtobufSentence: Protobuf Sentence this object represents."""

        self.sentence_set = None
        """SentenceSet: (Parent of this object.) List of sentences this sentence is contained in."""

        self.__index = 0
        """int: Index of the word to add to the parse tree."""

        self.parse_tree = ParseTree.fromprotobuf(
            self.parseTree,
            read_leaf=self.__read_leaf
        )
        """ParseTree: Parse tree of this sentence, altered so token and index of the word are added to each leaf."""

        self.basic_dependencies = DependencyGraph(self.basicDependencies, self)
        """DependencyGraph: Basic dependencies of the sentence."""

    def __repr__(self):
        return self.text

    @property
    def sentence(self) -> ProtobufSentence:
        """ProtobufSentence: Protobuf Sentence this object represents."""
        return str(self.index) + "_" + self._sentence

    @sentence.setter
    def sentence(self, sentence):
        self._sentence = sentence
        if sentence:
            self.tokens = TokenSet(self)
            self.mentions = NERMentionSet(self)

    @property
    def index(self) -> int:
        """Get the index of the sentence in its sentence set."""
        return self.sentence_set.index(self)

    @property
    def text(self) -> str:
        """Text of this sentence (from the original-text, no changed quotations)."""
        document = self.sentence_set.document
        return document.fulltext[self.tokens[0].beginChar:self.tokens[-1].endChar]

    @property
    def begin_char(self) -> int:
        """Beginning character number of the sentence."""
        return self.tokens[0].beginChar

    @property
    def end_char(self) -> int:
        """Ending character number of the sentence."""
        return self.tokens[-1].endChar

    def __read_leaf(self, leaf):
        leaf = ParseTree.Leaf(leaf)
        leaf.index = self.__index
        leaf.token = self.tokens[self.__index]
        self.__index += 1
        return leaf
