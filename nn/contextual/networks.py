from .gate import Gate
from .gru_layer import ContextualGRULayer
import lasagne
import lasagne.nonlinearities as nonlinearities
from nn.lasagne_nlp.networks.networks import build_BiGRU

def build_Attention_ContexualGRU(layer_char, layer_word, layer_lex, context, num_units, mask=None, grad_clipping=0, precompute_input=True,
                        peepholes=False, num_filters=[6, 14], dropout=True, in_to_out=False, filter_size=[7, 5]):
    # first get some necessary dimensions or parameters
    _, sent_length, _ = layer_word.output_shape

    # dropout before cnn?
    if dropout:
        layer_char = lasagne.layers.DropoutLayer(layer_char, p=0.5)

    layer_word_lex = lasagne.layers.concat([layer_word, layer_lex], axis=2)

    attention_vector = build_Attention_CNN(layer_char, sent_length, num_filters, filter_size)

    # finally, concatenate the two incoming layers together.
    gru_incoming = lasagne.layers.concat([layer_word_lex, attention_vector], axis=2)

    gru_context = lasagne.layers.concat([context, layer_lex, attention_vector], axis=2)

    return build_ContextualGRU(gru_incoming, gru_context, num_units, mask=mask, grad_clipping=grad_clipping,
                               precompute_input=precompute_input, dropout=dropout, in_to_out=in_to_out)

def build_Attention_CNN(layer_char, sent_length, num_filters, filter_size):

    # construct convolution layer
    cnn_layer = lasagne.layers.Conv1DLayer(layer_char, num_filters=num_filters[0], filter_size=filter_size[0],
                                           pad='full',
                                           nonlinearity=lasagne.nonlinearities.rectify, name='cnn', untie_biases=True)

    # cnn_layer = lasagne.layers.DimshuffleLayer(cnn_layer, pattern=(0, 2, 1))

    # infer the pool size for pooling (pool size should go through all time step of cnn)
    _, _, pool_size = cnn_layer.output_shape

    # construct max pool layer
    pool_layer = lasagne.layers.MaxPool1DLayer(cnn_layer, pool_size=pool_size)

    cnn_layer_1 = lasagne.layers.Conv1DLayer(pool_layer, num_filters=num_filters[1], filter_size=filter_size[1],
                                           pad='full',
                                           nonlinearity=lasagne.nonlinearities.rectify, name='cnn', untie_biases=True)

    cnn_layer_1 = lasagne.layers.DimshuffleLayer(cnn_layer_1, pattern=(0, 2, 1))

    # infer the pool size for pooling (pool size should go through all time step of cnn)
    _, _, pool_size = cnn_layer_1.output_shape
    #
    # # construct max pool layer
    pool_layer_1 = lasagne.layers.MaxPool1DLayer(cnn_layer_1, pool_size=pool_size)

    attention_vector = lasagne.layers.reshape(pool_layer_1, (-1, sent_length, [1]))

    return attention_vector

def build_character_embeddings(layer_char, sent_length, num_filters, filter_size):

    # construct convolution layer
    cnn_layer = lasagne.layers.Conv1DLayer(layer_char, num_filters=num_filters[0], filter_size=filter_size[0],
                                           pad='full',
                                           nonlinearity=lasagne.nonlinearities.rectify, name='cnn', untie_biases=True)

    # infer the pool size for pooling (pool size should go through all time step of cnn)
    _, _, pool_size = cnn_layer.output_shape

    # construct max pool layer
    pool_layer = lasagne.layers.MaxPool1DLayer(cnn_layer, pool_size=pool_size)

    cnn_layer_1 = lasagne.layers.Conv1DLayer(pool_layer, num_filters=num_filters[1], filter_size=filter_size[1],
                                             pad='full',
                                             nonlinearity=lasagne.nonlinearities.rectify, name='cnn', untie_biases=True)

    # infer the pool size for pooling (pool size should go through all time step of cnn)
    _, _, pool_size = cnn_layer_1.output_shape

    # construct max pool layer
    pool_layer_1 = lasagne.layers.MaxPool1DLayer(cnn_layer_1, pool_size=pool_size)

    # reshape the layer to match lstm incoming layer [batch * sent_length, num_filters, 1] --> [batch, sent_length, num_filters]
    output_cnn_layer = lasagne.layers.reshape(pool_layer_1, (-1, sent_length, [1]))

    return output_cnn_layer

def build_ContextualGRU(incoming, context, num_units, mask=None, grad_clipping=0, precompute_input=True, dropout=True, in_to_out=False):
    # dropout for incoming
    if dropout:
        incoming = lasagne.layers.DropoutLayer(incoming, p=0.5)

    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    resetgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_tid=lasagne.init.GlorotUniform(),
                             W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))

    updategate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_tid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.Uniform(range=0.1))
    # now use tanh for nonlinear function of hidden gate
    hidden_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None, W_tid=lasagne.init.GlorotUniform(),
                          nonlinearity=nonlinearities.tanh)

    gru_forward = ContextualGRULayer(incoming, context, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                          precompute_input=precompute_input,
                                          resetgate=resetgate_forward, updategate=updategate_forward,
                                          hidden_update=hidden_forward, name='forward')

    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    resetgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_tid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))

    updategate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_tid=lasagne.init.GlorotUniform(),
                               W_cell=lasagne.init.Uniform(range=0.1))
    # now use tanh for nonlinear function of hidden gate
    hidden_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None, W_tid=lasagne.init.GlorotUniform(),
                           nonlinearity=nonlinearities.tanh)

    gru_backward = ContextualGRULayer(incoming, context, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                           precompute_input=precompute_input, backwards=True,
                                           resetgate=resetgate_backward, updategate=updategate_backward,
                                           hidden_update=hidden_backward, name='backward')

    # concatenate the outputs of forward and backward GRUs to combine them.
    concat = lasagne.layers.concat([gru_forward, gru_backward], axis=2, name="bi-gru")

    # concat = gru_forward

    # dropout for output
    if dropout:
        concat = lasagne.layers.DropoutLayer(concat, p=0.5)

    if in_to_out:
        concat = lasagne.layers.concat([concat, incoming], axis=2)

    # the shape of BiRNN output (concat) is (batch_size, input_length, 2 * num_hidden_units)
    return concat

def build_Attention_GRU(layer_char, layer_word, num_units, mask=None, grad_clipping=0,
                        precompute_input=True,
                        peepholes=False, num_filters=[6, 14], dropout=True, in_to_out=False,
                        filter_size=[7, 5]):

    # first get some necessary dimensions or parameters
    _, sent_length, _ = layer_word.output_shape

    # dropout before cnn?
    if dropout:
        layer_char = lasagne.layers.DropoutLayer(layer_char, p=0.5)

    # layer_word_lex = lasagne.layers.concat([layer_word, layer_lex], axis=2)

    attention_vector = build_Attention_CNN(layer_char, sent_length, num_filters, filter_size)

    # finally, concatenate the two incoming layers together.
    gru_incoming = lasagne.layers.concat([layer_word, attention_vector], axis=2)

    return build_BiGRU(gru_incoming, num_units, mask=mask, grad_clipping=grad_clipping,
                       precompute_input=precompute_input, dropout=dropout, in_to_out=in_to_out)

def build_Lexicon_GRU(layer_word, layer_lex, num_units, mask=None, grad_clipping=0,
                        precompute_input=True,
                        dropout=True, in_to_out=False,
                        ):


    # first get some necessary dimensions or parameters
    _, sent_length, _ = layer_word.output_shape

    layer_word_lex = lasagne.layers.concat([layer_word, layer_lex], axis=2)

    return build_BiGRU(layer_word_lex, num_units, mask=mask, grad_clipping=grad_clipping,
                       precompute_input=precompute_input, dropout=dropout, in_to_out=in_to_out)

