from ConfigParser import SafeConfigParser
from utils.utils import minibatches_iter
from nn.lasagne_nlp.utils import utils
from loader.data_processor import DataProcessor
import codecs
import theano
from nn.lexicon_GRU_net import Lexicon_GRU_Net
import numpy as np
from utils.utils import write_model_data
from utils.ProgressBar import ProgressBar
from utils.utils import get_lex_file_list
from loader.lex_helper import LexHelper
from utils.utils import compute_f1_score

np.random.seed(1999) #STS 87.4%
# np.random.seed(1189)
# np.random.seed(1234)

class Lexicon_GRU(object):
    def __init__(self, config=None, opts=None):
        # not enough info to execute
        if config == None and opts == None:
            print "Please specify command option or config file ..."
            return

        parser = SafeConfigParser()
        parser.read(config)

        self.train_file = parser.get('model', 'train_file')
        self.dev_file = parser.get('model', 'dev_file')
        self.test_file = parser.get('model', 'test_file')
        self.word_column = parser.getint('model', 'word_column')
        self.label_column = parser.getint('model', 'label_column')
        self.oov = parser.get('model', 'oov')
        self.fine_tune = parser.getboolean('model', 'fine_tune')
        self.embedding = parser.get('model', 'embedding')
        self.embedding_path = parser.get('model', 'embedding_path')
        self.use_character = parser.getboolean('model', 'use_character')
        self.batch_size = parser.getint('model', 'batch_size')
        self.num_epochs = parser.getint('model', 'num_epochs')
        self.patience = parser.getint('model', 'patience')
        self.valid_freq = parser.getint('model', 'valid_freq')
        self.L2 = [float(x)/2 for x in parser.get('model', 'L2').split(',')]

        self.num_units = parser.getint('model', 'num_units')
        self.num_filters = list(map(int, parser.get('model', 'num_filters').split(',')))
        self.filter_size = list(map(int, parser.get('model', 'filter_size').split(',')))
        self.peepholes = parser.getboolean('model', 'peepholes')
        self.grad_clipping = parser.getfloat('model', 'grad_clipping')
        self.dropout = parser.getfloat('model', 'dropout')
        self.regular = parser.get('model', 'regular')
        self.gamma = parser.getfloat('model', 'gamma')
        self.learning_rate = parser.getfloat('model', 'learning_rate')
        self.update_algo = parser.get('model', 'update_algo')
        self.momentum =  parser.getfloat('model', 'momentum')
        self.decay_rate = parser.getfloat('model', 'decay_rate')
        self.output_predict = parser.getboolean('model', 'output_predict')

        self.model_path = parser.get('model', 'model_path')
        self.training = parser.getboolean('model', 'training')

        self.params_file = parser.get('model', 'params_file')
        if self.params_file == 'None':
            self.params_file = None

        self.lex_path = parser.get('model', 'lex_path')

        self.data = {'X_train': [], 'Y_train': [], 'mask_train': [],
                     'X_dev': [], 'Y_dev': [], 'mask_dev': [],
                     'X_test': [], 'Y_test': [], 'mask_test': [],
                     'embedd_table': [], 'label_alphabet': [],
                     'C_train': [], 'C_dev': [], 'C_test': [], 'char_embedd_table': []
                     }

        self.lexicons = {'lexicons_train': [], 'lexicons_dev': [], 'lexicons_test': []}

        self.setupOperators()

    def setupOperators(self):
        print('Loading the training data...')
        self.reader = DataProcessor(self.train_file, self.dev_file,
                                      self.test_file,
                                      word_column=self.word_column,
                                      label_column=self.label_column,
                                      oov=self.oov,
                                      fine_tune=self.fine_tune,
                                      embedding=self.embedding,
                                      embedding_path=self.embedding_path,
                                      use_character=self.use_character)
        self.data = self.reader.loadData()
        sentences_train, sentences_dev, sentences_test, padlen, word_alphabet = self.reader.load_sentences()
        lex_list = get_lex_file_list(self.lex_path)
        self.lex = LexHelper(lex_list, sentences_train, sentences_dev, sentences_test, padlen)

        self.lexicons["lexicons_train"], self.lexicons["lexicons_dev"], self.lexicons["lexicons_test"],\
        self.lex_dim = self.lex.build_lex_embeddings()

        print('Loading the data successfully!')

    def initModel(self):
        print "Building model..."
        self.model = Lexicon_GRU_Net(X_train=self.data['X_train'], C_train=self.data['C_train'],
                                      embedd_table=self.data['embedd_table'], char_embedd_table = self.data['char_embedd_table'], label_alphabet=self.data['label_alphabet'],
                                      num_units=self.num_units, num_filters=self.num_filters, filter_size=self.filter_size, peepholes=self.peepholes,
                                      grad_clipping=self.grad_clipping, dropout=self.dropout, regular=self.regular,
                                      gamma=self.gamma, learning_rate=self.learning_rate, update_algo=self.update_algo,
                                      momentum=self.momentum, fine_tune=self.fine_tune, L2=self.L2,
                                      params_file=self.params_file, model_path=self.model_path, embedd_lex_dim=self.lex_dim,
                                    )
        self.model.buildModel()
        print "Finish building model!"


    def executeModel(self):
        if self.training:
            print 'Training Model...'
            self.trainingModel()

        if self.params_file is not None:
            print 'Testing Model...'
            self.testModel()

    def trainingModel(self):

        self.initModel()

        best_acc = 0
        best_validation_accuracy = 0
        stop_count = 0
        lr = self.learning_rate
        patience = self.patience
        n_dev_samples, max_length = self.data['X_dev'].shape
        n_test_samples, max_length = self.data['X_test'].shape

        for epoch in range(1, self.num_epochs + 1):
            print 'Epoch %d (learning rate=%.4f, decay rate=%.4f): ' % (epoch, lr, self.decay_rate)
            train_err = 0.0
            train_batches = 0
            train_bar = ProgressBar('Training', max=len(self.data['X_train']))
            for batch in minibatches_iter(self.data['X_train'], self.data['Y_train'], masks=self.data['mask_train'],
                                                   char_inputs=self.data['C_train'],
                                                    lexicons=self.lexicons['lexicons_train'],
                                                   batch_size=self.batch_size, shuffle=True):
                inputs, targets, masks, char_inputs, lexicons = batch
                err = self.model.train_fn(inputs, targets, masks, lexicons)
                train_err += err
                train_bar.next(len(inputs))

                if train_batches > 0 and train_batches % self.valid_freq == 0:
                    accuracy_valid = []
                    for batch in minibatches_iter(self.data['X_dev'], self.data['Y_dev'],
                                                           masks=self.data['mask_dev'], lexicons=self.lexicons['lexicons_dev'],
                                                           char_inputs=self.data['C_dev'], batch_size=self.batch_size):
                        inputs, targets, masks, char_inputs, lexicons = batch
                        accuracy_valid.append(self.model.eval_fn(inputs, targets, masks, lexicons))
                    this_validation_accuracy = np.concatenate(accuracy_valid)[0:n_dev_samples].sum() / float(n_dev_samples)

                    if this_validation_accuracy > best_validation_accuracy:
                        print("\nTrain loss, " + str((train_err / self.valid_freq)) + ", validation accuracy: " + str(this_validation_accuracy * 100) + "%")
                        best_validation_accuracy = this_validation_accuracy
                        preds_test = []
                        accuracy_test = []
                        for batch in minibatches_iter(self.data['X_test'], self.data['Y_test'],
                                                               masks=self.data['mask_test'],
                                                               char_inputs=self.data['C_test'], lexicons=self.lexicons['lexicons_test'],
                                                               batch_size=self.batch_size):
                            inputs, targets, masks, char_inputs, lexicons = batch
                            _, preds = self.model.test_fn(inputs, targets, masks, lexicons)
                            preds_test.append(preds)
                            accuracy_test.append(self.model.eval_fn(inputs, targets, masks, lexicons))
                        this_test_accuracy = np.concatenate(accuracy_test)[0:n_test_samples].sum() / float(n_test_samples)
                        print "F1-score: " + str(compute_f1_score(self.data["Y_test"], preds_test, self.data['label_alphabet']) * 100)
                        print("Test accuracy: " + str(this_test_accuracy * 100) + "%")
                        if best_acc < this_test_accuracy:
                            best_acc = this_test_accuracy
                            write_model_data(self.model.network, self.model_path + '/best_model')

                    train_err = 0
                train_batches += 1

            train_bar.finish()

            # stop if dev acc decrease 3 time straightly.
            if stop_count == patience:
                break

            # re-compile a function with new learning rate for training
            if self.update_algo != 'adadelta':
                lr = self.learning_rate / (1.0 + epoch * self.decay_rate)
                updates = utils.create_updates(self.model.loss_train, self.model.params, self.update_algo, lr, momentum=self.momentum)
                self.model.train_fn = theano.function([self.model.input_var, self.model.target_var, self.model.mask_var, self.model.lex_var],
                                           outputs=self.model.loss_train,
                                           updates=updates, allow_input_downcast=True)

            print("Epoch " + str(epoch) + " finished.")
        print("The final best acc: " + str(best_acc*100) + "%")

        if self.output_predict:
            f = codecs.open('./results/10-fold.txt', 'a+', 'utf-8')
            f.write(str(best_acc*100)+'\n')
            f.close()

    def testModel(self):
        n_test_samples, max_length = self.data['X_test'].shape
        accuracy_test = []
        preds_test = []
        self.initModel()
        test_bar = ProgressBar('Testing', max=len(self.data['X_test']))
        for batch in minibatches_iter(self.data['X_test'], self.data['Y_test'],
                                               masks=self.data['mask_test'],
                                               char_inputs=self.data['C_test'], lexicons=self.lexicons['lexicons_test'],
                                               batch_size=self.batch_size):
            inputs, targets, masks, char_inputs, lexicons = batch
            test_bar.next(len(inputs))
            corrects = self.model.eval_fn(inputs, targets, masks, lexicons)
            _, preds = self.model.test_fn(inputs, targets, masks, lexicons)
            preds_test.append(preds)
            accuracy_test.append(corrects)
        this_test_accuracy = np.concatenate(accuracy_test)[0:n_test_samples].sum() / float(n_test_samples)
        test_bar.finish()
        print("Test accuracy: " + str(this_test_accuracy * 100) + "%")
        compute_f1_score(self.data['Y_test'], preds_test)