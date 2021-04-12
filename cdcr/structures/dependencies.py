from cdcr.structures.wrapper import Wrapper
from cdcr.structures.one_to_n_set import OneToNSet
from stanfordnlp.protobuf import DependencyGraph as ProtobufDependencyGraph
from typing import Any

ROOT = 'ROOT'


class Dependency(Wrapper):
    WRAPPED_ATTRIBUTE = "node"
    INDEX_OFFSET = 1
    """
    int: Offset between a node and the index (position) of a token in the sentence. Normally the
    node index is one higher than the index of the token in the sentence.
    """

    def __init__(self, node, dependency_graph):
        """
        A dependency of a token to another inside a sentence.

        Args:
            node: Node this Dependency represents.
            dependency_graph: Dependency graph this dependency is contained in.
        """
        self.dependency_graph = dependency_graph
        self.node = node

    @property
    def token(self):
        index = self.index - self.INDEX_OFFSET
        return self.dependency_graph.sentence.tokens[index]

    @property
    def is_root(self):
        return self.dependency_graph.root is self

    @property
    def edge(self):
        try:
            return self.dependency_graph.edge_by_target[self.index]
        except KeyError as e:
            if self.is_root:
                return None
            raise e

    @property
    def dep(self):
        try:
            if self.is_root:
                return ROOT
            return self.edge.dep
        except AttributeError as e:
            raise Exception(e)

    @property
    def offset_governor(self):
        if self.is_root:
            return 0
        return self.edge.source

    @property
    def governor(self):
        return self.offset_governor - self.INDEX_OFFSET

    @property
    def governor_token(self):
        if self.is_root:
            return None
        return self.dependency_graph.sentence.tokens[self.governor]

    @property
    def governor_gloss(self):
        if self.is_root:
            return ROOT
        return self.governor_token.word

    @property
    def offset_dependent(self):
        if self.is_root:
            return self.index
        return self.edge.target

    @property
    def dependent(self):
        return self.offset_dependent - self.INDEX_OFFSET

    @property
    def dependent_token(self):
        return self.dependency_graph.sentence.tokens[self.dependent]

    @property
    def dependent_gloss(self):
        return self.dependent_token.word

    @property
    def dict(self):
        return {
            "governor": self.governor,
            "governorGloss": self.governor_gloss,
            "dependent": self.dependent,
            "dependentGloss": self.dependent_gloss,
            "dep": self.dep
        }

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}{{{self.governor_gloss} ({self.governor}) → " \
               f"{self.dependent_gloss} ({self.dependent})}}"


class DependencyGraph(OneToNSet, Wrapper):

    CHILDRENS_PARENT_ATTRIBUTE = "dependency_graph"
    WRAPPED_ATTRIBUTE = "dependency_graph"

    def __init__(self, dependency_graph, sentence):
        super().__init__(dependency_graph.node)
        self.sentence = sentence
        self.dependency_graph = dependency_graph
        self.dependencies = []
        self.__cached_node_by_index = None
        self.__cached_edge_by_target = None

    @property
    def root_index(self):
        return self.dependency_graph.root[0]

    @property
    def root(self):
        return self.node_by_index[self.root_index]

    @property
    def node_by_index(self):
        if self.__cached_node_by_index:
            return self.__cached_node_by_index

        nodes = {}  # Indices starting at 1 -> dict.
        for node in self:
            nodes[node.index] = node

        self.__cached_node_by_index = nodes
        return nodes

    @property
    def edge_by_target(self):
        if self.__cached_edge_by_target:
            return self.__cached_edge_by_target

        edges = {}  # Indices starting at 1 -> dict.
        for edge in self.edge:
            edges[edge.target] = edge

        self.__cached_edge_by_target = edges
        return edges

    @property
    def root_first(self):
        copy = self.copy()
        root = self.root
        del copy[root.index - root.INDEX_OFFSET]
        copy.insert(0, root)
        return copy


    def _test_and_parse(self, node: Any, force: bool = False) -> Dependency:
        """
        Adding that a documents gets parsed to a Document() if it is none already.

        See ``structures.one_to_n_set.OneToNSet._test_and_parse()``.
        """
        if type(node) is not Dependency:
            node = Dependency(node, self)
        return super()._test_and_parse(node, force, True, None)
