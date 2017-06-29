import h5py
import numpy as np


class ContextVecLoader(object):
    def __init__(self, context_file):
        self.context_file = context_file

    def loadContexts(self):
        f = h5py.File(self.context_file, 'r')
        ks = f.keys()

        return ks
