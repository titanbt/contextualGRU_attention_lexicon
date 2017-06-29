from nn.lasagne_nlp.utils.utils import load_word_embedding_dict
from nn.lasagne_nlp.utils.data_processor import build_embedd_table

class ContextHelper(object):

    def __init__(self, embedding='Context', embedding_path=None, word_alphabet=None):
        self.embedding = embedding
        self.embedding_path = embedding_path
        self.word_alphabet = word_alphabet

    def build_context(self):

        print 'Building Context vectors!'

        embedd_dict, embedd_dim, caseless = load_word_embedding_dict(self.embedding, self.embedding_path, self.word_alphabet)

        print 'Finish building Context vectors!'

        return build_embedd_table(self.word_alphabet, embedd_dict, embedd_dim, caseless)

