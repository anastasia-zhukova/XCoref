from nltk.tree import ParentedTree
from stanfordnlp.protobuf import ParseTree as ProtobufParseTree
from typing import Optional, Callable


class ParseTree(ParentedTree):
    """
    Parentend Tree representing a parse tree offering a method to parse from a protobuf parse tree from
    stanfordnlp.
    """

    class Node(object):
        def __init__(self, node: ProtobufParseTree):
            """
            A node of this parse tree, as nodes here save more information it is not simply a str.

            The __str__ representation is still the same as a leaf of a normal tree.

            Args:
                node: Subtree of the original protobuf parse tree.
            """
            self.value = node.value
            self.score = node.score
            self.index = node.yieldBeginIndex

        def __str__(self):
            # Only show the value (tag), e.g. NP.
            return self.value

        def __repr__(self):
            # Add the module and name structures.tree.ParseTree.Node(NP).
            cls = self.__class__
            return cls.__module__ + "." + cls.__name__ + "(" + self.__str__() + ")"

    class Leaf(Node):
        def __init__(self, leaf: ProtobufParseTree):
            """
            A leaf of this parse tree, as leafs here save more information it is not simply a str.

            The __str__ representation is still the same as a leaf of a normal tree.

            Args:
                leaf: Subtree (Leaf) of the original protobuf parse tree.
            """
            super().__init__(leaf)

        def __repr__(self):
            # Add the module and name structures.tree.ParseTree.Node(NP).
            cls = self.__class__
            return cls.__module__ + "." + cls.__name__ + "(" + str(self.index) + ", " + self.__str__() + ")"


    @classmethod
    def fromprotobuf(
            cls,
            tree: ProtobufParseTree,
            read_node: Optional[Callable]=None,
            read_leaf: Optional[Callable]=None
    ):
        """
        Read Stanfords CoreNLP Protobuf Parse Tree (stanfordnlp.protobuf.ParseTree) and parse it to a
        nltk ParentedTree with Nodes and Leafes containing value and score (if not overwritten by `read_node` or
        `read_leaf`).

        Args:
            tree: The original parse tree to parse to a nltk tree.
            read_node: Function to be called with following signature:

                    read_node(ProtobufParseTree) -> Any

                This overrides the default behaviour (``read_node = ParseTree.Node``) and saves the returned value
                in the node.
            read_leaf: Function to be called with following signature:

                    read_leaf(ProtobufParseTree) -> Any

                This overrides the default behaviour (``read_node = ParseTree.Leaf``) and saves the returned value
                in the leaf.

        Returns:
            The parsed tree.
        """
        childs = []
        for child in tree.child:
            childs.append(ParseTree.fromprotobuf(child, read_node, read_leaf))

        if not childs:
            if not read_leaf:
                read_leaf = cls.Leaf
            return read_leaf(tree)

        if not read_node:
            read_node = cls.Node
        return ParseTree(read_node(tree), childs)
