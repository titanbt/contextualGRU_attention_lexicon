from utils.commandParser import DLNLPOptParser
from model.bilstm_deepcnn import BILSTM_DeepCNN
from model.model_attention_tf import Model_Attention_tf
from model.bilstm import BILSTM
from model.cnn import CNN
from model.attention_contextualGRU import Attention_ContexualGRU
from model.attention_GRU import Attention_GRU
from model.lexicon_GRU import Lexicon_GRU
from model.contextualGRU import ContexualGRU
from model.attention_lex_GRU import Attention_Lex_GRU
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

if __name__ == '__main__':

    args = DLNLPOptParser()
    config = args.config

    if args.mode == 'attention_contextualGRU':
        attention_contexualGRU = Attention_ContexualGRU(config, args)
        attention_contexualGRU.executeModel()
    elif args.mode == 'attention_GRU':
        attention_GRU = Attention_GRU(config, args)
        attention_GRU.executeModel()
    elif args.mode == 'lexicon_GRU':
        lexicon_GRU = Lexicon_GRU(config, args)
        lexicon_GRU.executeModel()
    elif args.mode == 'contextualGRU':
        contextualGRU = ContexualGRU(config, args)
        contextualGRU.executeModel()