from collections import OrderedDict
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer
import nltk.data
import StringIO
import glob
import numpy
import operator
import os
import random
import re
import string
import sys
import re
import xml.dom.minidom
import xml.etree.ElementTree

def has_letter(s):
  return re.search('[a-zA-Z]', s) != None

def get_category_from_training_doc_regexp(filename):
  pattern = r'<CATEGORY>(.*?)</CATEGORY>'
  groups = re.findall(pattern, open(filename).read(), re.S)
  if len(groups) > 0:
    return groups[0].strip()
  else:
    return ''

def get_text_from_training_doc_regexp(filename):
  pattern = r'<TEXT>(.*?)</TEXT>'
  groups = re.findall(pattern, open(filename).read(), re.S)
  assert len(groups) == 1
  text = groups[0]
  pattern = r'<P>(.*?)</P>'
  groups = re.findall(pattern, text, re.S)
  if len(groups) == 0:
    paragraphs = text.split('\t')
    paragraphs = [' '.join(p.split()) for p in paragraphs]
    return '\n\t'.join(paragraphs)
  else:
    return '\n\t'.join(groups)

def get_sentence_from_training_doc_regexp(filename):
  document = get_text_from_training_doc_regexp(filename)
  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
  sentences = sent_detector.tokenize(document)
  tknzr = TweetTokenizer()
  tokenized_sentences = []
  for i in xrange(len(sentences)):
    tokens = tknzr.tokenize(sentences[i])
    tokenized_sentences.append(' '.join(tokens))
  return tokenized_sentences

def get_sentence_from_training_doc(filename):
  pass

def get_sentence_from_testing_doc(filename):
  dom = xml.dom.minidom.parse(filename)
  text = dom.getElementsByTagName('TEXT')
  assert len(text) == 1
  document = text[0].childNodes[0].data
  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
  sentences = sent_detector.tokenize(document)
  tknzr = TweetTokenizer()
  tokenized_sentences = []
  for i in xrange(len(sentences)):
    t = tknzr.tokenize(sentences[i])
    tokenized_sentences.append(' '.join(t))
  return tokenized_sentences

def main(unused_argv):
  files = glob.glob('/usr/local/google/home/lijian/headlines/ethz-nlp-headline/duc2004/docs/*/*')
  for f in files:
    try:
      sentences = get_sentence_from_testing_doc(f)
    except:
      continue
    sentences = [s.lower() + '\n' for s in sentences]
    b = os.path.basename(f)
    outf = os.path.join('/tmp/test/', b)
    open(outf, 'w').writelines(sentences)
  pass

if __name__ == '__main__':
  main(sys.argv)
