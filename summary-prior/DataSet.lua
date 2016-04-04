-- The torch support large object only for Tensor, so we need to load the
-- sentences as tensor.
require("torch")
require("Utils")

local DataSet = torch.class("DataSet")

-- The class to hold the training, developing, and testing data.
function DataSet:__init(config, sentence_file, word_vocab)
  if config.kSampleSize > 0 and config.kBatchSize > 0 then
    assert(config.kSampleSize >= config.kBatchSize)
  end
  local lines = Utils.ReadTrainingLines(config, sentence_file)
  local num_tokens = Utils.CountTokensFromTrainingLines(lines)
  local num_lines = #lines
  assert(num_tokens > 0)
  assert(num_lines > 0)
  self.tokens_ = torch.IntTensor(num_tokens)
  self.starts_ = torch.IntTensor(num_lines)
  self.lengths_ = torch.IntTensor(num_lines)
  self.features_ = torch.FloatTensor(num_lines, 3)
  self.scores_ = torch.FloatTensor(num_lines, 2)
  local token = 1
  local sentence = 1
  for k = 1, #lines do
    local line = lines[k]
    if (sentence > num_lines) then
      break
    end
    assert(token <= num_tokens)
    local parts = Utils.GetTokensFromTrainingLine(line)
    assert(#parts > 0)
    local size = #parts
    self.lengths_[sentence] = size
    self.starts_[sentence] = token
    for i = 1, size do
      assert(parts[i] ~= nil)
      self.tokens_[token] = word_vocab:word_id(parts[i])
      token = token + 1
    end
    local avg_count = DataSet._GetAverageWordCount(parts, word_vocab)
    local avg_category_count = DataSet._GetAverageCategoryCount(parts, word_vocab)
    local pos = Utils.GetPositionScore(line)
    local rouge1, rouge2 = Utils.GetRougeScores(line)
    self.scores_[sentence][1] = rouge1
    self.scores_[sentence][2] = rouge2
    self.features_[sentence][1] = pos
    self.features_[sentence][2] = avg_count
    self.features_[sentence][3] = avg_category_count
    sentence = sentence + 1
  end
  assert(num_lines + 1 == sentence)
  assert(num_tokens + 1 == token)
  self.config_ = config
  if config.useGPU == true then
    require("cutorch")
    require("cunn")
    require("cudnn")
    require("cunnx")
  end

  self.word_vocab_ = word_vocab
end

function DataSet._GetAverageWordCount(tokens, word_vocab)
  assert(#tokens > 0)
  local sum = 0
  for _, w in ipairs(tokens) do
    sum = sum + word_vocab:text_count(w)
  end
  return sum / #tokens
end

function DataSet._GetAverageCategoryCount(tokens, word_vocab)
  assert(#tokens > 0)
  local sum = 0
  for _, w in ipairs(tokens) do
    sum = sum + word_vocab:text_category_count(w)
  end
  return sum / #tokens
end

function DataSet:size()
  return self.lengths_:size(1)
end
