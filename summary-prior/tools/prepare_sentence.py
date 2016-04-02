from collections import OrderedDict
from nltk.tokenize import RegexpTokenizer
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
import xml.dom.minidom
import xml.etree.ElementTree

def get_rouge_stats(rouge_out_file):
  eval_stats = {}
  for l in open(rouge_out_file):
    line = l.strip()
    if line.find('Eval') < 0:
      continue
    parts = line.split()
    assert(len(parts) == 7)
    label = parts[1]
    eval_id = int(float(parts[3]))
    r = parts[4]
    p = parts[5]
    f = parts[6]
    if eval_id not in eval_stats:
      eval_stats[eval_id] = []
    eval_stats[eval_id].append(' '.join([label, r, p, f]))
  return eval_stats

def get_rouge_settings(rouge_xml_file):
  dom = xml.dom.minidom.parse(rouge_xml_file)
  settings = {}
  evals = dom.getElementsByTagName('EVAL')
  for e in evals:
    r = e.getElementsByTagName('PEER-ROOT')
    eval_id = int(e.getAttribute('ID'))
    peer_path = r[0].childNodes[0].nodeValue
    peers = e.getElementsByTagName('P')
    assert(len(peers) == 1)
    peer = peers[0].childNodes[0].nodeValue
    sentence_pos = int(peers[0].getAttribute('ID'))
    pos_from_filename = int(os.path.splitext(peer)[0].split('-')[1])
    assert(sentence_pos == pos_from_filename)
    peer = os.path.join(peer_path, peer)
    settings[eval_id] = '\t'.join([peer, str(sentence_pos)])
  return settings

def get_sentences(settings, stats):
  tknzr = TweetTokenizer()
  assert(len(settings) == len(stats))
  sentences = []
  for k in settings:
    assert(k in stats)
    setting = settings[k].split('\t')
    stat = sorted(stats[k])
    sentence_filename = setting[0]
    sentence_pos = setting[1]
    tokens = tknzr.tokenize(open(sentence_filename).read().lower())
    parts = []
    parts.append(' '.join(tokens))
    parts.append('POS ' + sentence_pos)
    parts.extend(stat)
    sentences.extend('\t'.join(parts) + '\n')
  return sentences

def align_rouge_xml_out():
  doc_files = glob.glob('/tmp/rouge/*')
  doc_files = sorted(doc_files)
  print doc_files
  retv = []
  for i in xrange(0, len(doc_files), 2):
    o = doc_files[i]
    x = doc_files[i + 1]
    of, oe = os.path.splitext(o)
    xf, xe = os.path.splitext(x)
    assert(of == xf)
    retv.append([x, o])
  return retv


def main(unused_argv):
  xml_out = align_rouge_xml_out()
  sentences = []
  for k in xml_out:
    x = get_rouge_settings(k[0])
    o = get_rouge_stats(k[1])
    s = get_sentences(x, o)
    sentences.extend(s)

  open('/tmp/sentences.txt', 'w').writelines(sentences)
  pass


if __name__ == '__main__':
  main(sys.argv)

