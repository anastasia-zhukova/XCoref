import logging
import os
import gdown

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder


class ElmoEmbedding(object):
    '''
    A wrapper class for the ElmoEmbedder of Allen NLP.
    '''

    def __init__(self, options_file, weight_file):
        logging.info('Loading Elmo Embedding module')

        if not os.path.exists(weight_file):
            output, _ = os.path.split(weight_file)
            gdown.download("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/"
                           "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5", output, quiet=False)
            gdown.download("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/"
                           "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json", output, quiet=False)
        self.embedder = ElmoEmbedder(options_file, weight_file)
        logging.info('Elmo Embedding module loaded successfully')

    def get_elmo_avg(self, sentence):
        '''
        This function gets a sentence object and returns and ELMo embeddings of
        each word in the sentences (specifically here, we average over the 3 ELMo layers).
        :param sentence: a sentence object
        :return: the averaged ELMo embeddings of each word in the sentences
        '''
        tokenized_sent = sentence.get_tokens_strings()
        embeddings = self.embedder.embed_sentence(tokenized_sent)
        output = np.average(embeddings, axis=0)

        return output
