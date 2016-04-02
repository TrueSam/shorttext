#!/bin/bash

cd ~/headlines/ethz-nlp-headline/rouge-1.5.5

files=`ls /tmp/rouge/*.xml`

for f in $files; do
  echo $f
  outfile=${f/xml/out}
  # echo $outfile
  ./ROUGE-1.5.5.pl -e data -n 1 -n 2 -x -a -d -s $f > $outfile
done
