from nn.lasagne_nlp.utils import data_processor
from nn.lasagne_nlp.utils.alphabet import Alphabet

class DataProcessor(object):
    def __init__(self, train_file, dev_file, test_file, word_column, label_column,
                 oov='embedding', fine_tune=False, embedding='glove', embedding_path=None, use_character=False):

        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.word_column = word_column
        self.label_column = label_column
        self.embedding = embedding
        self.embedding_path = embedding_path
        self.oov = oov
        self.fine_tune = fine_tune
        self.use_character = use_character

        self.data = {'X_train': [], 'Y_train': [], 'mask_train': [],
                     'X_dev': [], 'Y_dev': [], 'mask_dev': [],
                     'X_test': [], 'Y_test': [], 'mask_test': [],
                     'embedd_table': [], 'label_alphabet': [],
                     'C_train': [], 'C_dev': [], 'C_test': [], 'char_embedd_table': []
                     }


    def loadData(self):

        self.data['X_train'], self.data['Y_train'],self.data['mask_train'], \
        self.data['X_dev'], self.data['Y_dev'], self.data['mask_dev'], \
        self.data['X_test'], self.data['Y_test'], self.data['mask_test'], \
        self.data['embedd_table'], self.data['label_alphabet'], \
        self.data['C_train'], self.data['C_dev'], self.data['C_test'], \
        self.data['char_embedd_table'] = data_processor.load_dataset_sequence_labeling(self.train_file, self.dev_file,
                                                                                                  self.test_file,
                                                                                                  word_column=self.word_column,
                                                                                                  label_column=self.label_column,
                                                                                                  oov=self.oov,
                                                                                                  fine_tune=self.fine_tune,
                                                                                                  embedding=self.embedding,
                                                                                                  embedding_path=self.embedding_path,
                                                                                                  use_character=self.use_character)

        return self.data

    def load_sentences(self):
        word_alphabet = Alphabet('word')
        label_alphabet = Alphabet('senti')

        # read training data
        print "Reading data for lexicon embedding and context..."
        word_sentences_train, _, _, _ = data_processor.read_conll_sequence_labeling(
            self.train_file, word_alphabet, label_alphabet, self.word_column, self.label_column)

        word_sentences_dev, _, _, _ = data_processor.read_conll_sequence_labeling(
            self.dev_file, word_alphabet, label_alphabet, self.word_column, self.label_column)

        word_sentences_test, _, _, _ = data_processor.read_conll_sequence_labeling(
            self.test_file, word_alphabet, label_alphabet, self.word_column, self.label_column)

        padlen = self.max_length(word_sentences_train, word_sentences_dev, word_sentences_test)

        # close alphabets
        word_alphabet.close()
        label_alphabet.close()

        print "Finish loading data for lexicon embedding and context..."
        return word_sentences_train, word_sentences_dev, word_sentences_test, padlen, word_alphabet

    def max_length(self, sentences_train, sentences_dev, sentences_test):
        train_lengh = max(len(x) for x in sentences_train)
        dev_lengh = max(len(x) for x in sentences_dev)
        test_lengh = max(len(x) for x in sentences_test)
        return max(train_lengh, dev_lengh, test_lengh)




