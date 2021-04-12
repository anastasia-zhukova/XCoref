"""
This is a workaround for pickle to be able to load CoreNLP protobuf elements.

If an error alike the following occurs, it might be necessary that this class is a subclass
of a protobuf class. This class still needs to be importable on top level.

Error:

.. code-block:: python

    _pickle.PicklingError: Can't pickle <class 'CoreNLP_pb2.CorefMention'>: attribute lookup CorefMention on CoreNLP_pb2 failed

Solution: Adding following to this file:

.. code-block:: python

    CorefMention = CorefChain.CorefMention

"""
from stanfordnlp.protobuf.CoreNLP_pb2 import *

CorefMention = CorefChain.CorefMention
Node = DependencyGraph.Node
Edge = DependencyGraph.Edge
