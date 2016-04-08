"""Build the training and testing data from DUC dataset.

Run the script under the current directory, the script will build data at
data/build directory.

"""

from collections import OrderedDict
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer
import nltk.data
import StringIO
import glob
import numpy
import math
import operator
import os
import subprocess
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
  return '\n\t'.join(get_paragraph_from_training_doc_regexp(filename))

def get_paragraph_from_training_doc_regexp(filename):
  pattern = r'<TEXT>(.*?)</TEXT>'
  groups = re.findall(pattern, open(filename).read(), re.S)
  assert len(groups) == 1
  text = groups[0]
  pattern = r'<P>(.*?)</P>'
  groups = re.findall(pattern, text, re.S)
  if len(groups) == 0:
    paragraphs = text.split('\t')
    paragraphs = [' '.join(p.split()) for p in paragraphs]
    return paragraphs
  else:
    paragraphs = [' '.join(g.split()) for g in groups]
    return paragraphs

def get_sentence_from_training_doc_regexp(filename):
  document = get_text_from_training_doc_regexp(filename)
  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
  sentences = sent_detector.tokenize(document)
  tknzr = TweetTokenizer()
  tokenized_sentences = []
  for i in xrange(len(sentences)):
    tokens = tknzr.tokenize(sentences[i].strip())
    tokenized_sentences.append(' '.join(tokens))
  return tokenized_sentences

def get_sentence_from_training_doc(filename):
  pass

def get_sentence_from_testing_doc(filename):
  dom = xml.dom.minidom.parse(filename)
  text = dom.getElementsByTagName('TEXT')
  assert len(text) == 1
  document = ' '.join(text[0].childNodes[0].data.split())
  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
  sentences = sent_detector.tokenize(document)
  tknzr = TweetTokenizer()
  tokenized_sentences = []
  for i in xrange(len(sentences)):
    t = tknzr.tokenize(sentences[i])
    tokenized_sentences.append(' '.join(t))
  return tokenized_sentences

class Word:
  def __init__(self):
    self.count = 0
    self.category_count = {}

def build_vocabuary(data_dir):
  doc_files = glob.glob(os.path.join(data_dir,
                                     'duc2003/docs.without.headlines/*/*'))

  assert len(doc_files) > 0
  tknzr = TweetTokenizer()
  vocabulary = {}
  for doc_file in doc_files:
    content = get_text_from_training_doc_regexp(doc_file)
    category = get_category_from_training_doc_regexp(doc_file)

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

  build_data_dir = os.path.join(data_dir, 'build')
  try:
    os.mkdir(build_data_dir)
  except OSError:
    print 'Directory %s already exists' % build_data_dir

  f = open(os.path.join(build_data_dir, 'vocabulary.txt'), 'w')
  x = sorted(vocabulary.items(), key=lambda x: x[1].count, reverse=True)
  for id in xrange(len(x)):
    if not has_letter(x[id][0]):
      continue

    fields = []
    fields.append(str(id))
    fields.append(x[id][0])
    v = x[id][1]
    fields.append(str(v.count))
    for c in v.category_count:
      fields.append('%s:%d' % (c, v.category_count[c]))
    f.write('\t'.join(fields) + '\n')

def align_doc_and_summaries(doc_files, summary_files):
  retv = {}
  for df in doc_files:
    dfb = os.path.basename(df)
    matched = []
    for sf in summary_files:
      sfb = os.path.basename(sf)
      if sfb.find(dfb) >= 0:
        matched.append(sf)
    retv[df] = matched
  return retv

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

def align_rouge_xml_out(model_dir):
  doc_files = glob.glob(os.path.join(model_dir, 'rouge-*'))
  doc_files = sorted(doc_files)
  assert len(doc_files) % 2 == 0
  retv = []
  for i in xrange(0, len(doc_files), 2):
    o = doc_files[i]
    x = doc_files[i + 1]
    of, oe = os.path.splitext(o)
    xf, xe = os.path.splitext(x)
    assert(of == xf)
    retv.append([x, o])
  return retv

def build_training_data(data_dir):
  doc_files = glob.glob(os.path.join(data_dir, 'duc2003/docs.without.headlines/*/*'))
  summary_files = glob.glob(os.path.join(data_dir, 'duc2003/detagged.duc2003.abstracts/models/*'))

  model_dir = os.path.join(data_dir, 'build/model')
  try:
    os.mkdir(model_dir)
  except OSError:
    print 'Directory %s already exists' % model_dir

  aligned_files = align_doc_and_summaries(doc_files, summary_files)
  rouge = []
  max_ps = 0
  for k in aligned_files:
    doc_file = k
    summary_files = aligned_files[k]

    ps = []
    ps = get_sentence_from_training_doc_regexp(doc_file)
    max_ps = max(max_ps, len(ps))
    b = os.path.basename(doc_file)
    paragraphs_files = []
    for i in xrange(len(ps)):
      f = os.path.join(model_dir, '%s-%03d.html' % (b, i))
      open(f, 'w').writelines([ps[i]])
      paragraphs_files.append(f)
    rouge.append([summary_files, paragraphs_files])

  # at most 100 paragraphs in the document
  for paragraph in xrange(max_ps):
    eval_id = 0
    rouge_lines = []
    for k in rouge:
      assert len(k) == 2
      assert len(k[0]) > 0
      assert len(k[1]) > 0
      paragraphs_files = k[1]
      summary_files = k[0]
      if len(paragraphs_files) <= paragraph:
        continue
      paragraph_file = paragraphs_files[paragraph]
      rouge_lines.append('<EVAL ID="%d">\n' % eval_id)
      rouge_lines.append('  <MODEL-ROOT>%s</MODEL-ROOT>\n' % os.path.dirname(summary_files[0]))
      rouge_lines.append('  <PEER-ROOT>%s</PEER-ROOT>\n' % os.path.dirname(paragraph_file))
      rouge_lines.append('  <INPUT-FORMAT TYPE="SPL">  </INPUT-FORMAT>\n')

      rouge_lines.append('  <PEERS>\n')
      rouge_lines.append('    <P ID="%d">%s</P>\n' % (paragraph, os.path.basename(paragraph_file)))
      rouge_lines.append('  </PEERS>\n')

      rouge_lines.append('  <MODELS>\n')
      for i in xrange(len(summary_files)):
        b = os.path.basename(summary_files[i])
        parts = b.split('.')
        rouge_lines.append('  <M ID="%s">%s</M>\n' % (parts[4], b))

      rouge_lines.append('  </MODELS>\n')
      rouge_lines.append('</EVAL>\n')

      eval_id = eval_id + 1
    if len(rouge_lines) > 0:
      rouge_config = open(os.path.join(model_dir, 'rouge-%03d.xml' % paragraph), 'w')
      rouge_config.write('<ROUGE_EVAL version="1.55">\n')
      rouge_config.writelines(rouge_lines)
      rouge_config.write('</ROUGE_EVAL>\n')

  rouge_xml_files = glob.glob(os.path.join(model_dir, 'rouge-*.xml'))
  assert len(rouge_xml_files) > 0
  rouge_dir = os.path.join(data_dir, '../rouge-1.5.5')
  for xml_file in rouge_xml_files:
    assert len(xml_file) > 0
    base, ext = os.path.splitext(xml_file)
    assert ext == '.xml'
    outfile = base + '.out'
    args = [os.path.join(rouge_dir, 'ROUGE-1.5.5.pl'), '-e', os.path.join(rouge_dir, 'data'),  '-n', '1', '-n', '2', '-x', '-a', '-d', '-s', xml_file]
    cmd = ' '.join(args) + ' > ' + outfile
    os.system(cmd)
    # p = os.popen(cmd, 'r')
    # p = subprocess.check_output(args, shell=True)
    # open(outfile, 'w').write(p.read())

  xml_out = align_rouge_xml_out(model_dir)
  sentences = []
  for k in xml_out:
    # Sometimes the xml parsing get strange errors.
    try:
      x = get_rouge_settings(k[0])
    except:
      continue
    o = get_rouge_stats(k[1])
    if len(o) == 0:
      continue
    s = get_sentences(x, o)
    sentences.extend(s)
  open(os.path.join(data_dir, 'build/sentences.txt'), 'w').writelines(sentences)

def build_testing_data(data_dir):
  test_data_dir = os.path.join(data_dir, 'build/test/')
  try:
    os.mkdir(test_data_dir)
  except OSError:
    print 'Directory %s already exists' % test_data_dir

  files = glob.glob(os.path.join(data_dir, 'duc2004/docs/*/*'))
  for f in files:
    try:
      sentences = get_sentence_from_testing_doc(f)
    except:
      continue
    sentences = [s.lower() + '\n' for s in sentences]
    b = os.path.basename(f)
    outf = os.path.join(test_data_dir, b)
    open(outf, 'w').writelines(sentences)

def main(unused_argv):
  data_dir = '../data'
  build_vocabuary(data_dir)
  build_training_data(data_dir)
  build_testing_data(data_dir)
  validate_dir = os.path.join(data_dir, 'build/validate')
  try:
    os.mkdir(validate_dir)
  except OSError:
    print 'Directory %s already exists' % validate_dir

if __name__ == '__main__':
  main(sys.argv)

