from contextlib import suppress

import graphene
from graphene import ObjectType, String, List, Field, Float, Int, Date, InputObjectType
from stanfordnlp.protobuf.CoreNLP_pb2 import Document, Sentence

from cdcr.structures.political_bias import PoliticalSide
from cdcr.structures.graphene_schema import SCHEMA
from cdcr.structures.sentiment import Sentiment

from cdcr.survey.survey import QuestionTypes


# class CorefMentionSchema(ObjectType):
#     mention = Field(SCHEMA(Mention))
#     is_representative = Boolean()
#     text = String()


# class CorefSchema(ObjectType):
#     mentions = List(CorefMentionSchema)
#     representative = CorefMentionSchema
#     coref = Field(SCHEMA(CorefChain))


class SentenceSchema(ObjectType):
    sentence = Field(SCHEMA(Sentence))
    text = String()
    begin_char = Int()
    end_char = Int()


class FulltextOffset(ObjectType):
    attribute = String()
    offset = Int()


class BiasCategory(ObjectType):
    category = String()
    probability = Float()


PoliticalSideSchema = graphene.Enum.from_enum(PoliticalSide)

QuestionTypesSchema = graphene.Enum.from_enum(QuestionTypes)


class QuestionSchema(ObjectType):
    key = String()
    text = String()
    answers = List(String)
    type = Field(QuestionTypesSchema)


class AnswerSchema(InputObjectType):
    question_key = String()
    answer = String()


class FrontendViews(graphene.Enum):
    START = 'start',
    OVERVIEW = 'overview',
    ARTICLE_VIEW = 'article_view',
    END = 'end',


class AnswerReturnSchema(ObjectType):
    next_view = Field(FrontendViews)
    next_iteration = String()


class PoliticalBiasSchema(ObjectType):
    side = Field(PoliticalSideSchema)
    probability = Float()


class DocumentSchema(ObjectType):
    title = String()
    description = String()
    text = String()
    sentences = List(SentenceSchema)
#     corefs = List(CorefSchema)
    document = Field(SCHEMA(Document))
    fulltext = String()
    representativeness = Float()
    index = Int()
    political_bias = Field(PoliticalBiasSchema)
    fulltext_offsets = List(FulltextOffset)

    def resolve_political_bias(root, info):
        return root.dummy_political_bias

    #def resolve_representativeness(root, info):
    #    return root.dummy_representativeness

    def resolve_fulltext_offsets(root, info):
        return [
            FulltextOffset(attribute=attribute, offset=offset)
            for attribute, offset in root.fulltext_offsets.items()]

    def resolve_title(root, info):
        with suppress(AttributeError):
            return root.title
        return None

    def resolve_description(root, info):
        with suppress(AttributeError):
            return root.description
        return None


SentimentSchema = graphene.Enum.from_enum(Sentiment)


class SentimentMentionSchema(ObjectType):
    sentiment = Field(SentimentSchema)
    probability = Field(Float)
    bias_categories = List(BiasCategory)

    def resolve_bias_categories(root, info):
        return [BiasCategory(category=key, probability=value) for key, value in root.bias_categories.items()]


class CandidateSchema(ObjectType):
    text = String()
    representative_word = String()
    sentiment = Field(SentimentMentionSchema)
    document = Field(DocumentSchema)
    begin_char = Int()
    end_char = Int()
    head_begin_char = Int()
    head_end_char = Int()


class EntitySchema(ObjectType):
    class SentimentPercentage(ObjectType):
        sentiment = Field(SentimentSchema)
        percentage = Float()
    members = List(CandidateSchema)
    representative = String()
    representative_phrases = List(String)
    sentiment_percentage = List(SentimentPercentage)

    def resolve_sentiment_percentage(root, info):
        sentiment_percentages = []
        for sentiment, percentage in root.sentiment_percentage.items():
            sentiment_percentages.append(
                EntitySchema.SentimentPercentage(sentiment=sentiment, percentage=percentage)
            )
        return sentiment_percentages


class DocumentSetSchema(ObjectType):
    topic = String()
    documents = List(DocumentSchema, indices=List(Int))
    entities = List(EntitySchema)
    candidates = List(List(CandidateSchema))

    #def __init__(self, document_set, filter):
    #    super().__init__(**{
    #        'topic': document_set
    #    })

    def resolve_candidates(root, info):
        return [
            list(group.values())[0] for group in root.candidates
        ]

    def resolve_documents(root, info, indices=None):
        if isinstance(root, DocumentSetSchema):
            root = root.documents
        if indices is not None:
            return [root[index] for index in indices]
        return root


class IdentifierDocumentSet(ObjectType):
    document_set = Field(DocumentSetSchema)
    identifier = Field(String)
    date = Date()
