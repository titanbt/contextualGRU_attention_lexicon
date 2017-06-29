import lasagne
import theano.tensor as T
import theano
from nn.contextual.networks import build_Lexicon_GRU
import lasagne.nonlinearities as nonlinearities
from lasagne_nlp.utils import utils
from utils.utils import read_model_data

class Lexicon_GRU_Net(object):
    def __init__(self, X_train, C_train, embedd_table=None, char_embedd_table = None, label_alphabet=None,
                 num_units=50, num_filters=[6,14], filter_size=[7,5], peepholes=False, grad_clipping=3, dropout=0.5,
                 regular=None, gamma=1e-6, learning_rate=0.1, update_algo='adadelta', momentum=0.9, fine_tune=True, L2=None, params_file=None, model_path=None,
                 embedd_lex_dim=15):

        self.X_train = X_train
        self.embedd_table = embedd_table
        self.char_embedd_table = char_embedd_table
        self.fine_tune = fine_tune
        self.label_alphabet = label_alphabet
        self.C_train = C_train
        self.num_units = num_units
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.peepholes = peepholes
        self.grad_clipping = grad_clipping
        self.dropout = dropout
        self.regular = regular
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.update_algo = update_algo
        self.momentum = momentum
        self.L2 = L2
        self.params_file = params_file
        self.model_path = model_path
        self.embedd_lex_dim = embedd_lex_dim


    def buildModel(self):

        def construct_input_layer():
            if self.fine_tune:
                layer_input = lasagne.layers.InputLayer(shape=(None, max_length), input_var=self.input_var, name='input')
                layer_embedding = lasagne.layers.EmbeddingLayer(layer_input, input_size=alphabet_size,
                                                                output_size=embedd_dim,
                                                                W=self.embedd_table, name='embedding')
                return layer_embedding
            else:
                layer_input = lasagne.layers.InputLayer(shape=(None, max_length, embedd_dim), input_var=self.input_var,
                                                        name='input')
                return layer_input

        def construct_lexicon_input_layer():
            layer_lex_embedding = lasagne.layers.InputLayer(shape=(None, max_length, self.embedd_lex_dim), input_var=self.lex_var, name='lex_input')
            return layer_lex_embedding

        num_labels = self.label_alphabet.size() - 1

        self.target_var = T.ivector(name='targets')
        self.mask_var = T.matrix(name='masks', dtype=theano.config.floatX)
        self.lex_var = T.tensor3(name='lexicons')

        if self.fine_tune:
            self.input_var = T.imatrix(name='inputs')
            num_data, max_length = self.X_train.shape
            alphabet_size, embedd_dim = self.embedd_table.shape
        else:
            self.input_var = T.tensor3(name='inputs', dtype=theano.config.floatX)
            num_data, max_length, embedd_dim = self.X_train.shape

        layer_word = construct_input_layer()
        layer_lex = construct_lexicon_input_layer()

        layer_mask = lasagne.layers.InputLayer(shape=(None, max_length), input_var=self.mask_var, name='mask')

        # construct model
        model = build_Lexicon_GRU(layer_word, layer_lex, self.num_units, mask=layer_mask,
                                    grad_clipping=self.grad_clipping,
                                    dropout=self.dropout)

        # construct output layer (dense layer with softmax)
        self.network = lasagne.layers.DenseLayer(model, num_units=num_labels, nonlinearity=nonlinearities.softmax,
                                                 name='softmax')

        if self.params_file is not None:
            print "Initializing params from file: %s" % self.params_file
            read_model_data(self.network, self.model_path, self.params_file)

        # get output of bi-lstm-cnn shape=[batch * max_length, #label]
        prediction_train = lasagne.layers.get_output(self.network)
        prediction_eval = lasagne.layers.get_output(self.network, deterministic=True)
        final_prediction = T.argmax(prediction_eval, axis=1)
        correct_predictions = T.eq(final_prediction, self.target_var)

        # compute loss
        self.loss_train = lasagne.objectives.categorical_crossentropy(prediction_train, self.target_var).mean()

        # l2 regularization?
        if self.regular == 'l2':
            l2_penalty = lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l2)
            self.loss_train = self.loss_train + self.gamma * l2_penalty

        l2_layers = []
        for layer in lasagne.layers.get_all_layers(self.network):
            if isinstance(layer, (lasagne.layers.EmbeddingLayer, lasagne.layers.Conv1DLayer, lasagne.layers.DenseLayer)):
                l2_layers.append(layer)
        self.loss_train = lasagne.objectives.aggregate(lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(self.network), self.target_var), mode='mean') + lasagne.regularization.regularize_layer_params_weighted(dict(zip(l2_layers, self.L2)), lasagne.regularization.l2)

        # hyper parameters to tune: learning rate, momentum, regularization.
        self.learning_rate = 1.0 if self.update_algo == 'adadelta' else self.learning_rate

        self.params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = utils.create_updates(self.loss_train, self.params, self.update_algo, self.learning_rate, momentum=self.momentum)

        # Compile a function performing a training step on a mini-batch
        self.train_fn = theano.function([self.input_var, self.target_var, self.mask_var, self.lex_var],
                                   outputs=self.loss_train,
                                   updates=updates, allow_input_downcast=True)
        # Compile a second function evaluating the loss and accuracy of network
        self.eval_fn = theano.function([self.input_var, self.target_var, self.mask_var, self.lex_var],
                                  outputs=correct_predictions, allow_input_downcast=True)

        self.test_fn = theano.function([self.input_var, self.target_var, self.mask_var, self.lex_var],
                                    outputs=[correct_predictions, final_prediction], allow_input_downcast=True)