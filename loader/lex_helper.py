import numpy as np
import re
from nn.lasagne_nlp.utils import data_processor
from nn.lasagne_nlp.utils.alphabet import Alphabet
import os

class LexHelper(object):
    def __init__(self, lex_path_list, sentences_train, sentences_dev, sentences_test, padlen):

        self.lex_path_list = lex_path_list
        self.sentences_train = sentences_train
        self.sentences_dev = sentences_dev
        self.sentences_test = sentences_test
        self.padlen = padlen

    def build_lex_embeddings(self):
        print 'Building lexicon embeddings'

        train, _ = self.pad_sentences(self.sentences_train, self.padlen)
        dev, _ = self.pad_sentences(self.sentences_dev, self.padlen)
        test, _ = self.pad_sentences(self.sentences_test, self.padlen)

        norm_model, raw_model = self.load_lexicon_unigram(self.lex_path_list)
        lexdim = 0
        for model_idx in range(len(norm_model)):
            lexdim += len(norm_model[model_idx].values()[0])

        lexiconModel = norm_model

        def get_index_of_vocab_lex(lexiconModel, word):
            lexiconList = np.empty([0, 1])
            for index, eachModel in enumerate(lexiconModel):
                if word in eachModel:
                    temp = np.array(np.float32(eachModel[word]))
                else:
                    temp = np.array(np.float32(eachModel["<PAD/>"]))
                lexiconList = np.append(lexiconList, temp)

            if len(lexiconList) > 16:
                print len(lexiconList)
                print '======================over 15======================'
            return lexiconList

        train_lex = np.array([[get_index_of_vocab_lex(lexiconModel, word) for word in sentence] for sentence in train])
        dev_lex = np.array([[get_index_of_vocab_lex(lexiconModel, word) for word in sentence] for sentence in dev])
        test_lex = np.array([[get_index_of_vocab_lex(lexiconModel, word) for word in sentence] for sentence in test])

        print 'Finish building lexicon embeddings!'

        return train_lex, dev_lex, test_lex, lexdim

    def load_lexicon_unigram(self, file_path_list):
        default_vector_dic = {'EverythingUnigramsPMIHS.txt': [0],
                              'HS-AFFLEX-NEGLEX-unigrams.txt': [0, 0, 0],
                              'Maxdiff-Twitter-Lexicon_0to1.txt': [0.5],
                              'S140-AFFLEX-NEGLEX-unigrams.txt': [0, 0, 0],
                              'unigrams-pmilexicon.txt': [0, 0, 0],
                              'unigrams-pmilexicon_sentiment_140.txt': [0, 0, 0],
                              'BL.txt': [0],
                              'ts-lex.txt': [0]}

        raw_model = [dict() for x in range(len(file_path_list))]
        norm_model = [dict() for x in range(len(file_path_list))]

        for index, each_model in enumerate(raw_model):
            data_type = file_path_list[index].replace("./resources/lexicons/", "")
            default_vector = default_vector_dic[data_type]

            # print data_type, default_vector
            raw_model[index]["<PAD/>"] = default_vector

            with open(file_path_list[index], 'r') as document:
                for line in document:
                    line_token = re.split(r'\t', line)

                    data_vec = []
                    key = ''

                    for idx, tk in enumerate(line_token):
                        if idx == 0:
                            key = tk
                        else:
                            try:
                                data_vec.append(float(tk))
                            except:
                                pass

                    assert (key != '')
                    each_model[key] = data_vec

        for index, each_model in enumerate(norm_model):
            # for m in range(len(raw_model)):
            values = np.array(raw_model[index].values())
            new_val = np.copy(values)

            # print 'model %d' % index
            for i in range(len(raw_model[index].values()[0])):
                pos = np.max(values, axis=0)[i]
                neg = np.min(values, axis=0)[i]
                mmax = max(abs(pos), abs(neg))
                # print pos, neg, mmax

                new_val[:, i] = values[:, i] / mmax

            keys = raw_model[index].keys()
            dictionary = dict(zip(keys, new_val))

            norm_model[index] = dictionary

        return norm_model, raw_model

    def pad_sentences(self, sentences, padlen, padding_word="<PAD/>"):
        """
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        """
        if padlen == None:
            sequence_length = max(len(x) for x in sentences)
        else:
            sequence_length = padlen

        padded_sentences = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        return padded_sentences, sequence_length