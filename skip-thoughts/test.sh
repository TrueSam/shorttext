#!/usr/bin/env bash

set -e

files=`find . -name '*_test.lua'`
for f in $files; do
  echo $f
  $HOME/torch/install/bin/th $f
done

