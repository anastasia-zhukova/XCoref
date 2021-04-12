import gensim.models.keyedvectors as word2vec


# If we call word2vec models from the web app, we need to use threading instead of the ModelLoader.
# For more info read https://stackoverflow.com/questions/42986405/how-to-speed-up-gensim-word2vec-model-load-time
# def init():
#     model = gensim.models.KeyedVectors.load_word2vec_format(config.WORD2VEC_MODEL_PATH,
#                                                             binary=True)
#     model.init_sims(replace=True)
#     model.save('./resources/word_vector_models/GoogleNews-vectors-negative300/GoogleNews-vectors-gensim-normed.bin')
#     model = KeyedVectors.load
#     ('./resources/word_vector_models/GoogleNews-vectors-negative300/GoogleNews-vectors-gensim-normed.bin', mmap='r')
#     model.syn0norm = model.syn0  # prevent recalc of normed vectors
#     model.most_similar('stuff')  # any word will do: just to page all in
#     Semaphore(0).acquire()  # just hang until process killed
#1
#     _models = {}
#
#
def init():
    global _models
    _models = {}


def load(path):
    if path in _models:
        return _models[path]

    model = word2vec.KeyedVectors.load_word2vec_format(path, binary=True)
    _models[path] = model
    return model


def unload(path):
    del _models[path]
