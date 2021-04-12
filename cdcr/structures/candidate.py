import shortuuid
from nltk.corpus import wordnet as wn
from typing import *

from cdcr.structures.sentiment import SentimentMention
from cdcr.structures.coref_mention_set import CorefMention
from cdcr.candidates.cand_enums import *
from cdcr.structures.sentence import Sentence
from cdcr.structures.token import Token
from cdcr.structures.document import Document
from cdcr.structures.dependencies import Dependency


class Candidate:
    """
    A class to store text excerpts as candidates for framing devices.
    """

    def __init__(self, text: str,
                 sentence: Sentence,
                 tokens: List[Token],
                 is_representative: bool,
                 cand_id: str = None,
                 head_token: Token = None,
                 cand_type: CandidateType = CandidateType.NP,
                 coref_subtype: str = None,
                 change_head: ChangeHead = ChangeHead.NON_NUMBER,
                 enhancement: ExtentedPhrases = ExtentedPhrases.EXTENDED,
                 origin_type: OriginType = OriginType.EXTRACTED,
                 annot_label=None,
                 annot_type=None,
                 **kwargs):
        """

        Args:
            text: A text of the mention
            sentence: A sentence from which the mention is extracted
            tokens: Tokens of the mention
            is_representative: If a mention is a representative phrase in its candidate group. If a mention is a part of
                coreference chain, then this property is extracted from the chain, if not, then the mention is a
                representative phrase by default.
            cand_id: ID of the mention; can be internally generated
            head_token: A head token of the phrase
            cand_type: A type of the candidate, which can be set to CandidateType.COREF, CandidateType.NP,
                CandidateType.VP, or CandidateType.OTHER (if headtoken is not NP or VP)
            coref_subtype: A candidate subtype is usually a type that comes from a coreference chain
            change_head: A strategy that changes headword in phrases with numbers, e.g., "millions of immigrants",
                from "millions" to "immigrants"; can be set to ChangeHead.ORIG and the head will remain unchanged.
            enhancement: A strategy that extends a NP if there is a longer NP with the same headword at the same position
                available, e.g., a NP "Trump" can be extended to "Donald Trump"
            origin_type: Origin of a candidate, i.e., it can be automatically extracted, obtained from human annotations,
                or combine both approaches.
            annot_label: A manually assigned label of an entity to which this phrase is referring, e.g., "Donald Trump"
            annot_type: A type of the manually assigned label, e.g., "Person"
            **kwargs: other parameters obtained from different dataset annotations
        """

        self.sentence = sentence
        self.max_num_words_cand = sentence.sentence_set.document.document_set.configuration.cand_extraction_config.max_phrase_len

        params = "".join([str(self.document.id) + str(sentence.index) + text +
                   "".join([str(v) for v in list(kwargs.values())]) if kwargs is not None else ""])

        self.id = shortuuid.uuid(name=params) if cand_id is None else cand_id

        self.text = text
        self.original_text = text
        self.is_representative = is_representative

        self.type = cand_type
        self.coref_subtype = coref_subtype
        self.enhancement = enhancement
        self.origin_type = origin_type

        self.tokens = tokens

        self.change_head_strategy = change_head

        # text can be != annot_text if candidate was extracted and then mapped to annotation
        self.annot_text = text if origin_type == OriginType.ANNOTATED else None
        self.annot_label = annot_label
        self.annot_type = annot_type

        self.parent_tree = None
        self.dependency_subtree = self.find_dep_subset()

        self.head_token = head_token if head_token is not None else self.find_head()
        self.head_token_original = self.head_token

        # additional attributes that we be assigned during the pipeline execution
        self.sentiment = None
        self.emotion_dimensions = {}
        self.emotion_group = None

        # apply strategies that modify original mention phrase and its properties
        if self.enhancement == ExtentedPhrases.EXTENDED:
            self.extend_phrase()

        if self.change_head_strategy == ChangeHead.NON_NUMBER:
            self.change_head_func()

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def document(self) -> Document:
        """
        Return a document from which this candidate was extracted

        Returns:
            A document object.

        """
        return self.sentence.sentence_set.document

    @property
    def begin_char(self) -> int:
        return self.tokens[0].beginChar

    @property
    def end_char(self) -> int:
        return self.tokens[-1].endChar

    @property
    def head_begin_char(self) -> int:
        return self.head_token.beginChar

    @property
    def head_end_char(self) -> int:
        return self.head_token.endChar

    def __repr__(self):
        return self.text.replace("_", " ")

    def __str__(self):
        return self.text.replace("_", " ")

    def is_extended(self) -> bool:
        """
        Checks of a phrase extension was applied.

        Returns:
            Boolean indicating if there was an extended NP to the original phrase.

        """
        return self.original_text != self.text

    def calculate_sentiment(self, classifier):
        """
        Calculate the sentiment of this candidate based on the classifier (of type TargetSentimentClassifier).

        """

        sentiment = classifier.infer(text=self.sentence.text,
                                     target_mention_from=self.tokens[0].sentence_begin_char,
                                     target_mention_to=self.tokens[-1].sentence_end_char)

        self.sentiment = SentimentMention.from_newstsc(sentiment[0])

    def set_emotion_dimension(self, category: str, values):
        """
        Sets emotion dimensions to the candidate.

        Args:
            category: A type of emotion dimensions.
            values: Values of emotion dimensions.

        """
        self.emotion_dimensions[category] = values

    def get_emotion_dimension(self, category: str):
        """
        Returns emotion dimesions for a given category.
        Args:
            category: A category of an emotion dimension.

        """
        return self.emotion_dimensions[category]

    def extend_phrase(self):
        """
        If there is a NP in the parsing tree that contains the current phrase, use it as a candidate phrase.

        """

        if self.head_token is None:
            return

        all_trees = list(self.sentence.parse_tree.subtrees())
        coref_index_set = set([token.index for token in self.tokens])

        dep_subtree = []
        if len(coref_index_set) > 1:
            for dep in self.sentence.basic_dependencies.root_first:
                if {dep.governor, dep.dependent}.issubset(coref_index_set):
                    dep_subtree.append(dep)

        for subtree in reversed(all_trees):
            subtree_set = set(list(j.index for j in subtree.leaves()))

            if coref_index_set.issubset(subtree_set) and len(coref_index_set) < len(subtree_set):
                new_dep_subtree = [dep for dep in self.sentence.basic_dependencies
                                   if dep.dependent in subtree_set and dep.governor in subtree_set]
                new_tokens = [self.sentence.tokens[j]
                                       for j in range(min(subtree_set), max(subtree_set) + 1)]
                new_head = self.find_head(new_tokens, new_dep_subtree)
                and_case_index = -1

                try:
                    if any([j.label().value == "CC" for j in subtree]):
                        and_case_index = [dep.dependent for dep in new_dep_subtree if dep.dep in ["cc", "conj"]][0]
                except IndexError:
                    pass

                if subtree.label().value.startswith("NP") and len(subtree.leaves()) <= self.max_num_words_cand:
                    form_text = False
                    if new_head == self.head_token and and_case_index == -1:
                        self.tokens = new_tokens
                        form_text = True
                    if and_case_index > -1:
                        if and_case_index > self.head_token.index:
                            if not len(self.sentence.tokens[new_tokens[0].index: and_case_index]):
                                continue
                            self.tokens = self.sentence.tokens[new_tokens[0].index: and_case_index]
                            self.dependency_subtree = [dep for dep in new_dep_subtree
                                                   if dep.dependent < and_case_index and dep.governor < and_case_index]
                        else:
                            if not len(self.sentence.tokens[and_case_index + 1: new_tokens[-1].index + 1]):
                                continue
                            self.tokens = self.sentence.tokens[and_case_index + 1: new_tokens[-1].index + 1]
                            self.dependency_subtree = [dep for dep in new_dep_subtree
                                                   if dep.dependent > and_case_index and dep.governor > and_case_index]
                        self.head_token = self.find_head()
                        form_text = True
                    if form_text:
                        self.text = "".join([t.word + t.after for t in self.tokens])
                        break
                if self.type == CandidateType.COREF:
                    break

    def find_dep_subset(self) -> List[Dependency]:
        """
        A caller of a static method find_dep_subset_static.

        Returns:
            A dependency subtree of the candidate.

        """
        return Candidate.find_dep_subset_static(self.tokens, self.sentence)

    @staticmethod
    def find_dep_subset_static(tokens: List[Token], sentence: Sentence) -> List[Dependency]:
        """
        Finds a dependency subtree of the candidate.

        Args:
            tokens: A list of candidate tokens.
            sentence: A sentence from which the phrase was extracted.

        Returns:
            A dependency subtree of the candidate.

        """
        dep_subtree = []
        coref_index_set = set([token.index for token in tokens])

        for dep in sentence.basic_dependencies:
            if {dep.governor, dep.dependent}.issubset(coref_index_set):
                dep_subtree.append(dep)

        return dep_subtree

    def find_head(self, tokens: List[Token] = None, dep_subtree: List[Dependency] = None) -> Token:
        """
        A caller of the static metod find_head_static.

        Args:
            tokens: A list of candidate's tokens.
            dep_subtree: A candidate's dependency subtree.

        Returns:
            A headword token.

        """
        if tokens is None or dep_subtree is None:
            tokens = self.tokens
            dep_subtree = self.dependency_subtree

        return Candidate.find_head_static(tokens, dep_subtree)

    @staticmethod
    def find_head_static(tokens: List[Token], dep_subtree: List[Dependency]) -> Token:
        """
        Find a headword token in a given list of tokens.

        Args:
            tokens: A list of candidate's tokens.
            dep_subtree: A candidate's dependency subtree.

        Returns:
            A headword token.

        """
        if not len(dep_subtree):
            return tokens[-1]

        gov_set = set()
        dep_set = set()
        for dep in dep_subtree:
            gov_set.add(dep.governor)
            dep_set.add(dep.dependent)

        root_list = list(gov_set.difference(dep_set))

        if not len(root_list):
            # should not happen
            return tokens[-1]

        head_index = list(root_list)[0]
        return {token.index: token for token in tokens}[head_index]

    def change_head_func(self):
        """
        Some phrases contain quantifiers as headwords, e.g., "thousands of people". The function replaces the
        identified head of phrase, e.g., "thousands", to the dependent word that contains the meaning of the phrase,
        e.g., "people".

        """

        def __check_quantity(quant_word):
            if quant_word.lower() in ["group", "groups"]:
                return True
            syn = wn.synsets(quant_word)
            for i_id, s in enumerate(syn):
                if s.lexname() == "noun.quantity" or "quantifier" in s.definition():  # , "noun.Tops"]:
                    return True
            return isinstance(quant_word, int)

        orig_head = self.head_token.word
        is_quantity = __check_quantity(orig_head)
        if not is_quantity:
            return

        new_head_index = None

        # check for cases like "hundreds of thousands"
        quant_words = [orig_head]
        for token in self.tokens:
            if __check_quantity(token.word) and token.word != orig_head and ("NN" in token.pos
                                                                          or token.pos == "CD" or token.pos == "JJ"):
                quant_words.append(token.word)

        match = 0
        for dep in self.dependency_subtree:
            if dep.dep == "nmod" and dep.governor_gloss in quant_words \
                    and dep.dependent_gloss not in quant_words:
                new_head_index = dep.dependent
                match += 1
            if dep.dep == "case" and dep.dependent_gloss == "of":
                match += 1
        if new_head_index is not None and match >= 2:
            self.head_token = {token.index: token for token in self.tokens}[new_head_index]

    @staticmethod
    def from_mention(mention: CorefMention,
                     enhancement: ExtentedPhrases = ExtentedPhrases.EXTENDED,
                     change_head: ChangeHead = ChangeHead.NON_NUMBER):
        """
        Create a CorefNPCandidate from a mention object.

        Args:
            mention: Mention to creat the CorefNPCandidate from.
            enhancement: A strategy that extends a NP if there is a longer NP with the same headword at the same position
                available, e.g., a NP "Trump" can be extended to "Donald Trump".
            change_head: A strategy that changes headword in phrases with numbers, e.g., "millions of immigrants",
                from "millions" to "immigrants"; can be set to ChangeHead.ORIG and the head will remain unchanged.

        Returns:
            A Candidate created from the data contained in the mention.
        """

        # Check if mention is in a document.
        try:
            document = mention.mention_set.coref.coref_set.document
        except AttributeError:
            raise ValueError("from_mention requires the passed mention to have "
                             "Mention.mention_set.coref.coref_set.document to be set.")

        candidate = Candidate(text=mention.text,
                              sentence=document.sentences[mention.sentenceIndex], tokens=mention.tokens,
                              is_representative=mention.is_representative, head_token=mention.head_token,
                              cand_type=CandidateType.COREF, coref_subtype=mention.mentionType,
                              enhancement=enhancement, change_head=change_head)

        return candidate
