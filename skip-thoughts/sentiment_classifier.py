from scilearn import svm
import sys
import os
import random

class Data:
    def __init(self, text, vector, label):
        self.text_ = text
        self.vector_ = vector
        self.label_ = label

class DataSet:
    def __init(self):
        self.data_ = []

    def AppendFromFile(text_filename, vector_filename, label):
        lines = open(text_filename).readlines()
        texts = [line.strip() for line in lines]
        lines = open(vector_filename).readlines()
        vectors = []
        for line in lines:
            v = [float(p) for p in line.strip().split()]
            vectors.append(v)
        assert(len(texts) == len(vectors))
        for i in xrange(len(texts)):
            d = Data(texts[i], vectors[i], label)
            self.data_.append(d)


if __name__ == '__main__':
    pass
