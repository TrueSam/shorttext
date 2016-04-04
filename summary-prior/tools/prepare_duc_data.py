import sys
import os
import glob
import xml.dom.minidom
import StringIO
import xml.etree.ElementTree
import duc_data

DUC_DOCS_PATH = ''
OUTPUT_PATH = ''

def get_paragraph_as_line(paragraph):
  sio = StringIO.StringIO(paragraph)
  lines = sio.readlines()
  retv = []
  for line in lines:
    line = line.strip()
    if len(line) > 0:
      retv.append(line)

  return ' '.join(retv)

def get_doc_content_as_paragraphs(filename):
  dom = xml.dom.minidom.parse(filename)
  paragraphs = dom.getElementsByTagName('P')
  if len(paragraphs) == 0:
    text = dom.getElementsByTagName('TEXT')
    assert len(text) == 1
    paragraphs = text[0].childNodes[0].data.split('\t')
  else:
    ps = []
    for p in paragraphs:
      ps.append(p.childNodes[0].nodeValue)
    paragraphs = ps

  retv = []
  for p in paragraphs:
    if len(p.strip()) > 0:
      retv.append(get_paragraph_as_line(p))

  return retv

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

def main(unused_argv):
  doc_files = glob.glob('/usr/local/google/home/lijian/headlines/ethz-nlp-headline/duc2003/docs.without.headlines/*/*')
  summary_files = glob.glob('/usr/local/google/home/lijian/headlines/ethz-nlp-headline/duc2003/detagged.duc2003.abstracts/models/*')
  reference_dir = '/tmp/reference'
  model_dir = '/tmp/model'
  aligned_files = align_doc_and_summaries(doc_files, summary_files)
  rouge = []
  for k in aligned_files:
    doc_file = k
    summary_files = aligned_files[k]

    ps = []
    ps = duc_data.get_paragraph_from_training_doc_regexp(doc_file)
#     try:
#       ps = get_doc_content_as_paragraphs(doc_file)
#     except:
#       continue
    b = os.path.basename(doc_file)
    paragraphs_files = []
    for i in xrange(len(ps)):
      f = os.path.join(model_dir, '%s-%03d.html' % (b, i))
      open(f, 'w').writelines([ps[i]])
      paragraphs_files.append(f)
    rouge.append([summary_files, paragraphs_files])

  # at most 100 paragraphs in the document
  for paragraph in xrange(100):
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
      rouge_config = open('/tmp/rouge/rouge-%03d.xml' % paragraph, 'w')
      rouge_config.write('<ROUGE_EVAL version="1.55">\n')
      rouge_config.writelines(rouge_lines)
      rouge_config.write('</ROUGE_EVAL>\n')

if __name__ == '__main__':
  main(sys.argv)
