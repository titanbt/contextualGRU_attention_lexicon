
import numpy

class DataReader(object):
    def __init__(self, train_file, train_label,
                 dev_file, dev_label,
                 test_file, test_label):
        self.data = {'X_train': [], 'Y_train': [], 'length_train': [],
                     'X_dev': [], 'Y_dev': [], 'length_dev': [],
                     'X_test': [], 'Y_test': [], 'length_test': []}

        self.mode = 'train'
        self._setupData(train_file, train_label, dev_file, dev_label, test_file, test_label)

    def readAll(self, mode='train'):
        return self.data['X_' + mode], \
               self.data['Y_' + mode], \
               self.data['length_' + mode]

    def _setupData(self, train_file, train_label,
                   dev_file, dev_label,
                   test_file, test_label):

        self.data['X_train'], self.data['Y_train'], self.data['length_train'] = self._loadData(train_file, train_label)
        self.data['X_dev'], self.data['Y_dev'], self.data['length_dev'] = self._loadData(dev_file, dev_label)
        self.data['X_test'], self.data['Y_test'], self.data['length_test'] = self._loadData(test_file, test_label)

    def _loadData(self, x_file, y_file, padding_value=15448):

        sorted_dict = {}
        x_data = []
        i=0
        file = open(x_file,"r")
        for line in file:
            words = line.split(",")
            result = []
            length=None
            for word in words:
                word_i = int(word)
                if word_i == padding_value and length==None:
                    length = len(result)
                result.append(word_i)
            x_data.append(result)

            if length==None:
                length=len(result)

            if length in sorted_dict:
                sorted_dict[length].append(i)
            else:
                sorted_dict[length]=[i]
            i+=1

        file.close()

        file = open(y_file,"r")
        y_data = []
        for line in file:
            words = line.split(",")
            y_data.append(int(words[0])-1)
        file.close()

        new_train_list = []
        new_label_list = []
        lengths = []
        for length, indexes in sorted_dict.items():
            for index in indexes:
                new_train_list.append(x_data[index])
                new_label_list.append(y_data[index])
                lengths.append(length)

        return numpy.asarray(new_train_list,dtype=numpy.int32),numpy.asarray(new_label_list,dtype=numpy.int32),lengths

    def pad_to_batch_size(self, array,batch_size):
        rows_extra = batch_size - (array.shape[0] % batch_size)
        if len(array.shape)==1:
            padding = numpy.zeros((rows_extra,),dtype=numpy.int32)
            return numpy.concatenate((array,padding))
        else:
            padding = numpy.zeros((rows_extra,array.shape[1]),dtype=numpy.int32)
            return numpy.vstack((array,padding))

    def extend_lenghts(self, length_list,batch_size):
        elements_extra = batch_size - (len(length_list) % batch_size)
        length_list.extend([length_list[-1]]*elements_extra)

