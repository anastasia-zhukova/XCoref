from cdcr.util.pb2_graphene import Protobuf2Graphene
from stanfordnlp.protobuf.CoreNLP_pb2 import DESCRIPTOR


SCHEMA = Protobuf2Graphene(DESCRIPTOR, lambda n: "Protobuf" + n.name)
