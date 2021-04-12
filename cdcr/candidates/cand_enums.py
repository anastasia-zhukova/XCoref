from enum import Enum


class CandidateType(Enum):
    """
    A candidate type.
    """
    COREF = 1
    NP = 2
    VP = 3
    OTHER = 4

    @classmethod
    def from_string(cls, text):
        return cls[text.upper()]


class OriginType(Enum):
    """
    Origin type of candidates.

    Extracted: automatically extracted from text.
    Annotated: obtained from a file with manual annotations .
    Extracted-Annotated: extracted phrases are the basis of the candidate corpus but they are expanded with annotated
    phrases if there are some non-matching phrases.
    """
    EXTRACTED = 1
    ANNOTATED = 2
    EXTRACTED_ANNOTATED = 3

    @classmethod
    def from_string(cls, text):
        return cls[text.upper()]


class CorefStrategy(Enum):
    """
    A coreference resolution scale.
    One_doc: coreferences are extracted from one text article.
    Multi_doc: coreferences are extracted from a group of articles, usually 5.
    No_coref: no coreference resolution needed, only independent phrases.
    """
    ONE_DOC = 1
    MULTI_DOC = 2
    NO_COREF = 3

    @classmethod
    def from_string(cls, text):
        return cls[text.upper()]


class ExtentedPhrases(Enum):
    """
    A phrase extension if a parent NP from a parse tree with the same head word is available.
    For example, "a beautiful house" instead of "house".
    As_is: no phrase extensing
    Extended: find the largest parent NP
    """
    ASIS = 1
    EXTENDED = 2

    @classmethod
    def from_string(cls, text):
        return cls[text.upper()]


class ChangeHead(Enum):
    """
    An option to specify if candidate extractor needs to change head of phrase in a phrase with quantifiers.
    For example, "_hundreds_ of birds" to "hundreds of _birds_"
    """
    ORIG = 1
    NON_NUMBER = 2

    @classmethod
    def from_string(cls, text):
        return cls[text.upper()]
