from typing import List
from cdcr.structures.token import Token
from enum import Enum


class Sentiment(Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'
    POSITIVE_AND_NEGATIVE = 'positive_and_negative'
    NEUTRAL = 'neutral'


class SentimentMention:
    """A mention expressing a sentiment."""
    def __init__(self, sentiment: Sentiment, reasoning_tokens: List[List[Token]], probability=None, bias_categories=None):
        if bias_categories is None:
            bias_categories = {}
        self.sentiment = sentiment
        self.reasoning_tokens = reasoning_tokens
        self.probability = probability
        self.bias_categories = bias_categories

    @staticmethod
    def from_newstsc(sentiment):
        return SentimentMention(
            Sentiment(sentiment["class_label"]),
            [],  # Currently not implemented into newstsc
            sentiment["class_prob"]
        )
