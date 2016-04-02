require("torch")
require("os")
require("Utils")
require("WordStat")
require('pl.utils')
require('pl.stringx')

local VocabularyUtils = torch.class("VocabularyUtils")

local function _SortedKeys(count)
  local items = {}
  for k, v in pairs(count) do
    local t = {k, v}
    table.insert(items, t)
  end
  table.sort(items, function(x, y) return x[2].count > y[2].count end)
  local keys = {}
  for i = 1, #items do
    table.insert(keys, items[i][1])
  end
  return keys
end

local function _FilterVocabulary(count, threshold)
  assert(threshold >= 0)
  local tmp = {}
  for k, v in pairs(count) do
    if v.count >= threshold then
      tmp[k] = v
    end
  end
  return tmp
end

local function _LoadCounts(in_file, lowercase)
  assert(in_file ~= nil and in_file:len() > 0)
  local count = {}
  for line in io.lines(in_file) do
    local parts = line:split('\t')
    assert(#parts >= 3, line)
    local i = tonumber(parts[1])
    local s = parts[2]
    local n = tonumber(parts[3])
    assert(n ~= nil)
    assert(i ~= nil)
    assert(s ~= nil)
    local w = WordStat()
    w.count = n
    w.category_count = #parts - 3
    count[s] = w
  end
  return count
end

local function _InitWordCount()
  local scored_count = {}
  local id
  local count_freq = {}

  scored_count[VocabularyUtils.kEndSentenceWordText] = VocabularyUtils.kEndSentenceWordId
  id = VocabularyUtils.kEndSentenceWordId + 1
  local w = WordStat()
  w.count = 0.0
  w.category_count = 0
  count_freq[VocabularyUtils.kEndSentenceWordId] = w

  scored_count[VocabularyUtils.kUnknownWordText] = VocabularyUtils.kUnknownWordId
  id = VocabularyUtils.kUnknownWordId + 1
  w = WordStat()
  w.count = 0.0
  w.category_count = 0
  count_freq[VocabularyUtils.kUnknownWordId] = w

  return scored_count, count_freq, id
end

local function _ScoreCount(unscored_count, scored_count, count_freq, next_id)
  local sorted_keys = _SortedKeys(unscored_count)
  for i = 1, #sorted_keys do
    local k = sorted_keys[i]
    local c = unscored_count[k]
    scored_count[k] = next_id
    count_freq[next_id] = c
    next_id = next_id + 1
  end
  for i = 1, next_id - 1 do
    assert(count_freq[i] ~= nil)
  end
  return scored_count, count_freq
end

-- The following are the predefined text and id for the word, characters, and
-- tag vocabulary.
VocabularyUtils.kEndSentenceWordText = "EOS"
VocabularyUtils.kEndSentenceWordId = 1
VocabularyUtils.kUnknownWordText = "UNK"
VocabularyUtils.kUnknownWordId = 2

VocabularyUtils.ReadWordVocabulary = function(filename, min_frequency, lowercase)
  local count = _LoadCounts(filename, lowercase)
  local unscored_count = _FilterVocabulary(count, min_frequency)
  local scored_count, count_freq, next_id = _InitWordCount()
  scored_count, count_freq = _ScoreCount(unscored_count, scored_count, count_freq, next_id)
  return scored_count, count_freq
end

function VocabularyUtils.PrintBatchSentences(word_vocab, sentences)
  assert(#sentences > 0)
  for j = 1, #sentences do
    local sentence = sentences[j]
    for i = 1, #sentence do
      assert(sentence[i]:size(1) == sentence[1]:size(1))
    end
    local batch_size = sentence[1]:size(1)
    assert(batch_size > 0)
    for k = 1, batch_size do
      local s = {}
      for i = 1, #sentence do
        table.insert(s, word_vocab:word_text(sentence[i][k]))
      end
      print(table.concat(s, ' '))
    end
    print('-------------------------------------------------')
  end
  print('=================================================')
end
