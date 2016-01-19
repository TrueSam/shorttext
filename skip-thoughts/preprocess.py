from collections import OrderedDict
from nltk.tokenize import RegexpTokenizer
import numpy
import operator
import os
import glob
import random
import string
import sys
import re

MAX_SENTENCE_LENGTH = -1

def ispunct(s):
    return all(c in string.punctuation for c in s)

def isnumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    output = []
    for i in xrange(0, len(l), n):
        output.append(l[i:i+n])

def cleanlines(lines):
    clean = []
    for cc in lines:
        words = cc.split()
        keep = []
        for w in words:
            if isnumber(w):
                w = '0' * len(w)
            if ispunct(w):
                continue
            keep.append(w)
        if len(keep) > 0:
            if MAX_SENTENCE_LENGTH > 0:
                clean.extend(chunks(keep, MAX_SENTENCE_LENGTH))
            else:
                clean.append(keep)
    output = []
    for c in clean:
        output.append(' '.join(c))
    return output

def cleantext(input_filepattern, output_file):
    lines = []
    files = glob.glob(input_filepattern)
    for f in files:
        for line in open(f):
            line = line.strip()
            if len(line) > 0:
                lines.append(line )
    lines = cleanlines(lines)
    lines = [line + '\n' for line in lines]
    f = open(output_file, 'w')
    f.writelines(lines)

def buildvocablary(input_file, output_file):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    wordcount = OrderedDict()
    for cc in open(input_file, 'r'):
        words = cc.strip().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1
    words = wordcount.keys()
    freqs = wordcount.values()
    sorted_idx = numpy.argsort(freqs)[::-1]

    f = open(output_file, 'w')
    worddict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        # The index starts from 1.
        f.write('%s\t%d\n' % (words[sidx], freqs[sidx])) # 1: <eos>, 2: <unk>

if __name__ == '__main__':
    # cleantext('../sent2vec/data/books_large_p?.txt', 'data/books.wrd')
    # buildvocablary('data/books.wrd', 'data/books.wrd.voc')
    # cleantext('../sent2vec/data/rt-polaritydata/rt-polarity.neg', 'data/rt.neg.wrd')
    # cleantext('../sent2vec/data/rt-polaritydata/rt-polarity.pos', 'data/rt.pos.wrd')
