import gdown
from typing import List, Set


from pymagnitude import *
from cdcr.config import *
from cdcr.structures.token import Token
from sklearn.metrics.pairwise import cosine_similarity as cs
from cdcr.entities.dict_lists import LocalDictLists
from cdcr.entities.const_dict_global import *


wordvectors = {
    WORD2VEC_WE: WORD2VEC_MAGN_PATH,
    FASTTEXT_WE: FASTTEXT_MAGN_PATH,
    ELMO_WE: ELMO_MAGN_PATH,
    GLOVE_WE: GLOVE_MAGN_PATH
}


class MagnitudeModel:

    def __init__(self, we_name, we_optimization=True):
        self.we_name = we_name
        if we_name == WORD2VEC_WE:
            if not os.path.isfile(wordvectors[we_name]):
                gdown.download("http://magnitude.plasticity.ai/word2vec/medium/GoogleNews-vectors-negative300.magnitude",
                               WORD2VEC_MAGN_PATH, quiet=False)
        elif we_name == FASTTEXT_WE:
            if not os.path.isfile(wordvectors[we_name]):
                gdown.download("http://magnitude.plasticity.ai/fasttext/medium/crawl-300d-2M.magnitude",
                               FASTTEXT_MAGN_PATH, quiet=False)
        elif we_name == ELMO_WE:
            if not os.path.isfile(wordvectors[we_name]):
                gdown.download("http://magnitude.plasticity.ai/elmo/medium/elmo_2x2048_256_2048cnn_1xhighway_weights.magnitude",
                               ELMO_MAGN_PATH, quiet=False)
        elif we_name == GLOVE_WE:
            if not os.path.isfile(wordvectors[we_name]):
                gdown.download("http://magnitude.plasticity.ai/glove/medium/glove.840B.300d.magnitude",
                               GLOVE_MAGN_PATH, quiet=False)
        self._model = Magnitude(wordvectors[we_name], stream=False)
        self._we_opt = we_optimization
        self.vector_size = self._model.dim

    def n_similarity(self, tokens_1: Union[List[Token], List[str]], tokens_2: Union[List[Token], List[str]]) -> float:
        if not len(tokens_1) or not len(tokens_2):
            return 0

        if type(tokens_1[0]) == str or type(tokens_2[0]) == str:
            word_vectors_1 = self._model.query(tokens_1)
            word_vectors_2 = self._model.query(tokens_2)

        else:
            if self.we_name == ELMO_WE:
                # get vectors for the full sentences first
                word_vectors_1 = self._elmo_vect(tokens_1)
                word_vectors_2 = self._elmo_vect(tokens_2)
            else:
                word_vectors_1 = self._model.query(self.optimize_phrase(tokens_1, self._we_opt))
                word_vectors_2 = self._model.query(self.optimize_phrase(tokens_1, self._we_opt))

        phrase_vector_1 = np.mean(word_vectors_1, axis=0)
        phrase_vector_2 = np.mean(word_vectors_2, axis=0)
        sim = cs([phrase_vector_1], [phrase_vector_2])
        return sim[0][0]

    def query(self, tokens: Union[List[Token], List[str]], coefs=None):
        if not len(tokens):
            return [np.zeros(self._model.dim)]

        if type(tokens[0]) == str:
            vectors = self._model.query(tokens)
        else:
            if self.we_name == ELMO_WE:
                vectors = self._elmo_vect(tokens)
            else:
                vectors = self._model.query([t.word for t in tokens])

        if coefs is None:
            return [np.mean(vectors, axis=0)]
        return [np.mean(np.array(coefs).reshape(-1,1) * vectors, axis=0)]

    def get_vector(self, word: str):
        return self._model.query(word)

    def _elmo_vect(self, tokens):
        token_indeces = [t.index for t in tokens]
        sent = tokens[0].token_set
        sent_vectors = self._model.query([t.word for t in sent])
        return sent_vectors[token_indeces]

    def optimize_phrase(self, tokens: Union[List[Token], Set[Token]], opt=False):
        """
        The method checks the words against a word embedding model and merges them into phrases if some word collocations
         are present in the model as a frequent phrase, e.g., "illegal_aliens"
        Args:
            tokens: list of tokens
            is_head: if the list of tokens is constructed our of head words

        Returns:
            A list of words as strings that are definitely present in the word embedding model.

        """

        representatives = []
        local_token_dict = {}

        for token in tokens:
            if token.word in LocalDictLists.stopwords or token.word in LocalDictLists.pronouns:
                continue

            if token.ner != NON_NER:
                representatives.append(token.word)
                local_token_dict[token.word] = token
            else:
                representatives.append(token.word.lower())
                local_token_dict[token.word.lower()] = token

        if not opt or self.we_name != WORD2VEC_WE:
            return list(set(representatives))

        if len(representatives) > 1:
            adj = []
            nn = []
            merged_words = representatives.copy()
            for repr_word in representatives:
                if bool(re.match(JJ, local_token_dict[repr_word].pos)):
                    adj.append(repr_word)
                if bool(re.match(NN, local_token_dict[repr_word].pos)):
                    nn.append(repr_word)

            for a in adj:
                for n in nn:
                    if a + "_" + n in self._model:
                        merged_words.remove(a)
                        merged_words.remove(n)
                        merged_words.append(a + "_" + n)
                        return merged_words
            for n_r in reversed(nn):
                for n in nn:
                    if n_r + "_" + n in self._model:
                        if n_r in merged_words:
                            merged_words.remove(n_r)
                        if n in merged_words:
                            merged_words.remove(n)
                        merged_words.append(n_r + "_" + n)
                        return merged_words
            for a_r in reversed(adj):
                for a in adj:
                    if a_r + "_" + a in self._model:
                        if a_r in merged_words:
                            merged_words.remove(a_r)
                        if a in merged_words:
                            merged_words.remove(a)
                        merged_words.append(a_r + "_" + a)
                        return merged_words
            return merged_words
        else:
            return representatives

