from collections import OrderedDict
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer
import StringIO
import glob
import numpy
import operator
import os
import random
import re
import string
import sys
import xml.dom.minidom
import xml.etree.ElementTree
import duc_data

class Word:
  def __init__(self):
    self.count = 0
    self.category_count = {}

def main(unused_argv):
  doc_files = glob.glob('/usr/local/google/home/lijian/headlines/ethz-nlp-headline/duc2003/docs.without.headlines/*/*')
  tknzr = TweetTokenizer()
  vocabulary = {}
  for doc_file in doc_files:
    content = duc_data.get_text_from_training_doc_regexp(doc_file)
    category = duc_data.get_category_from_training_doc_regexp(doc_file)

    tokens = tknzr.tokenize(content)
    for t in tokens:
      t = t.lower()
      if t not in vocabulary:
        vocabulary[t] = Word()
      w = vocabulary[t]
      w.count = w.count + 1
      if len(category) > 0:
        if category not in w.category_count:
          w.category_count[category] = 0
        w.category_count[category] = w.category_count[category] + 1

  f = open('/tmp/vocabulary.txt', 'w')
  x = sorted(vocabulary.items(), key=lambda x: x[1].count, reverse=True)
  for id in xrange(len(x)):
    if not duc_data.has_letter(x[id][0]):
      continue

    fields = []
    fields.append(str(id))
    fields.append(x[id][0])
    v = x[id][1]
    fields.append(str(v.count))
    for c in v.category_count:
      fields.append('%s:%d' % (c, v.category_count[c]))
    f.write('\t'.join(fields) + '\n')

if __name__ == '__main__':
  main(sys.argv)
