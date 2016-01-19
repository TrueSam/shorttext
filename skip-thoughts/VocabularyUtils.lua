require("torch")
require("Utils")

local utf8 = require("lua-utf8")

local VocabularyUtils = torch.class("VocabularyUtils")

local function _SortedKeys(count)
  local items = {}
  for k, v in pairs(count) do
    local t = {k, v}
    table.insert(items, t)
  end
  table.sort(items, function(x, y) return x[2] > y[2] end)
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
    if v >= threshold then
      tmp[k] = v
    end
  end
  return tmp
end

local function _LoadCounts(in_file, lowercase)
  assert(in_file ~= nil and in_file:len() > 0)
  local count = {}
  for line in io.lines(in_file) do
    local parts = Utils.split(line)
    assert(#parts == 2, line)
    local n = tonumber(parts[2])
    assert(n ~= nil)
    local s = parts[1]
    if lowercase == true then
      s = utf8.lower(s)
    end
    count[s] = n
  end
  return count
end

local function _InitWordCount()
  local scored_count = {}
  local id
  local count_freq = {}

  scored_count[VocabularyUtils.kEndSentenceWordText] = VocabularyUtils.kEndSentenceWordId
  id = VocabularyUtils.kEndSentenceWordId+ 1
  count_freq[VocabularyUtils.kEndSentenceWordId] = 1.0

  scored_count[VocabularyUtils.kUnknownWordText] = VocabularyUtils.kUnknownWordId
  id = VocabularyUtils.kUnknownWordId + 1
  count_freq[VocabularyUtils.kUnknownWordId] = 1.0

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
  local sum = 0.0
  for k, v in pairs(count_freq) do
    sum = sum + v
  end
  for k, v in pairs(count_freq) do
    count_freq[k] = v / sum
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
