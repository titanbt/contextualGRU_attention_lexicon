import numpy as np
import gzip
import theano
from gensim.models.word2vec import Word2Vec

class DataLoader(object):
    def __init__(self, tweet_file, test_file, vocab_file, val_ratio=0.05, train_ratio=1, embedding='glove', embedding_path=None):

        self.tweet_file = tweet_file
        self.test_file = test_file
        self.vocab_file = vocab_file
        self.val_ratio = val_ratio
        self.embedding = embedding
        self.embedding_path = embedding_path
        self.train_ratio = train_ratio

    def loadDataset(self):
        print ('Loading Data...')
        # train set and dev set will be divide based on val_ratio
        vmap = {}
        with open(self.vocab_file, "r") as vf:
            for line in vf:
                id, w, cnt = line.strip().decode('utf-8').split("\t")
                vmap[w] = int(id)

        label_map = {'negative': 0,
                     'positive': 1}

        def read_data(infile):
            X = []
            y = []
            nerrs = 0
            with open(infile, "r") as tf:
                for line in tf:
                    parts = line.strip().split('\t')
                    if len(parts) != 3:
                        nerrs += 1
                        continue
                    _, label, tweet = parts[0], parts[1], parts[2]
                    if label == "neutral":
                        continue
                    X.append(map(int, tweet.strip().split()))
                    y.append(label_map[label.strip()])
            print('%d bad lines for file %s' % (nerrs, infile))
            X = self._pad_mask(X)
            y = np.asarray(y, dtype=np.int32)
            return X, y

        X, y = read_data(self.tweet_file)
        n = X.shape[0]
        n_val = int(self.val_ratio * n)
        indices = np.arange(n)
        np.random.shuffle(indices)
        val = int((n - n_val)*self.train_ratio)
        train_indices = indices[:val]
        val_indices = indices[n - n_val:]

        X_test, y_test = read_data(self.test_file)

        embedd_dict, embedd_dim, _ = self._load_word_embedding_dict(self.embedding, self.embedding_path, vmap)
        embedd_table = self._build_embedding_table(vmap, embedd_dict, embedd_dim)

        print ('Load data successfully!')

        return (X[train_indices], y[train_indices],
                X[val_indices], y[val_indices],
                X_test, y_test,
                vmap, embedd_table, embedd_dim)

    def _build_embedding_table(self, vmap, embedd_dict, embedd_dim):
        V = len(vmap)
        print('Loading embeddings from file')
        print 'Done.'
        K = embedd_dim  # override dim
        print('Embedding dim: %d' % K)
        W = np.zeros((V, K), dtype=np.float32)
        no_vectors = 0
        for w in vmap:
            if w in embedd_dict:
                W[vmap[w]] = np.asarray(embedd_dict[w], dtype=np.float32)
            else:
                W[vmap[w]] = np.random.normal(scale=0.01, size=K)
                no_vectors += 1
        print "Initialized with word2vec. Couldn't find", no_vectors, "words!"
        return W

    def _load_word_embedding_dict(self, embedding, embedding_path, vmap, embedd_dim=100):

        if embedding == 'word2vec':
            # loading word2vec
            print("Loading word2vec ...")
            word2vec = Word2Vec.load_word2vec_format(embedding_path, binary=True)
            embedd_dim = word2vec.vector_size
            return word2vec, embedd_dim, False
        elif embedding == 'glove':
            # loading GloVe
            print("Loading GloVe ...")
            embedd_dim = -1
            embedd_dict = dict()
            with gzip.open(embedding_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    tokens = line.split()
                    if embedd_dim < 0:
                        embedd_dim = len(tokens) - 1
                    else:
                        assert (embedd_dim + 1 == len(tokens))
                    embedd = np.empty([1, embedd_dim], dtype=theano.config.floatX)
                    embedd[:] = tokens[1:]
                    embedd_dict[tokens[0]] = embedd
            return embedd_dict, embedd_dim, True
        elif embedding == 'senna':
            # loading Senna
            print("Loading Senna ...")
            embedd_dim = -1
            embedd_dict = dict()
            with gzip.open(embedding_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    tokens = line.split()
                    if embedd_dim < 0:
                        embedd_dim = len(tokens) - 1
                    else:
                        assert (embedd_dim + 1 == len(tokens))
                    embedd = np.empty([1, embedd_dim], dtype=theano.config.floatX)
                    embedd[:] = tokens[1:]
                    embedd_dict[tokens[0]] = embedd
            return embedd_dict, embedd_dim, True
        elif embedding == 'random':
            # loading random embedding table
            print("Loading Random ...")
            embedd_dict = dict()
            scale = np.sqrt(3.0 / embedd_dim)
            for word in vmap:
                embedd_dict[word] = np.random.uniform(-scale, scale, [1, embedd_dim])
            return embedd_dict, embedd_dim, False
        else:
            raise ValueError("embedding should choose from [word2vec, senna]")

    def _pad_mask(self, X, max_seq_length=140):

        N = len(X)
        X_out = np.zeros((N, max_seq_length, 2), dtype=np.int32)
        for i, x in enumerate(X):
            n = len(x)
            if n < max_seq_length:
                X_out[i, :n, 0] = x
                X_out[i, :n, 1] = 1
            else:
                X_out[i, :, 0] = x[:max_seq_length]
                X_out[i, :, 1] = 1

        return X_out