//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>

using namespace std;

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f;
  char file_name[max_size];
  float len;
  long long words, size, a, b;
  char ch;
  float *M;
  char *vocab;
  if (argc < 3) {
    printf("Usage: ./%s <INFILE> <OUTFILE>\nwhere INFILE contains word "
           "projections in the BINARY FORMAT\n", argv[0]);
    return 0;
  }
  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }

  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);

  string s = argv[2];
  size_t at = s.find('@');
  vector<string> output_filenames;
  if (at == string::npos) {
    output_filenames.push_back(argv[2]);
  } else {
    string base = s.substr(0, at);
    int shards = atoi(s.substr(at + 1).c_str());
    assert(shards > 0);
    for (int i = 0; i < shards; ++i) {
      sprintf(file_name, "%s-%05d-of-%05d", base.c_str(), i, shards);
      output_filenames.push_back(file_name);
    }
  }
  assert(output_filenames.size() > 0);
  vector<pair<long long, long long> > range(output_filenames.size());
  long long lenght;
  if (words % output_filenames.size() == 0) {
    lenght = words / output_filenames.size();
  } else {
    lenght = words / output_filenames.size() + 1;
  }
  assert(lenght > 0);
  for (int i = 0; i < output_filenames.size(); ++i) {
    long long start = i * lenght;
    long long end = start + lenght - 1;
    end = min(end, words);
    assert(start >= 0 && start < words);
    assert(end >= 0 && end < words);
    assert(start < end);

    strcpy(file_name, output_filenames[i].c_str());
    f = fopen(file_name, "w");
    if (f == NULL) {
      printf("Can't create output file\n");
      return -1;
    }
    fprintf(f, "%lld\n", end - start + 1);
    fprintf(f, "%lld\n", size);
    for (b = start; b < end; b++) {
      fprintf(f, "%s", &vocab[b * max_w]);
      for (a = 0; a < size; a++) {
        fprintf(f, " %g", M[a + b * size]);
      }
      fprintf(f, "\n");
    }
    fclose(f);
  }

  return 0;
}
