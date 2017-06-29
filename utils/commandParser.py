import argparse


def DLNLPOptParser():
    
    parser = argparse.ArgumentParser(\
            description='Default DLNLP opt parser.')

    parser.add_argument('-mode', default='attention_contextualGRU')
    parser.add_argument('-config',default='config/attention_context_s140_sr_w2v.cfg', help='config file to set.')

    return parser.parse_args()

