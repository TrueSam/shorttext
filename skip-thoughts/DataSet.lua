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
  local num_tokens, num_lines = Utils.CountTokensAndLines(config, sentence_file)
  assert(num_tokens > 0)
  assert(num_lines > 0)
  self.tokens_ = torch.IntTensor(num_tokens)
  self.starts_ = torch.IntTensor(num_lines)
  self.lengths_ = torch.IntTensor(num_lines)
  local token = 1
  local sentence = 1
  for line in io.lines(sentence_file) do
    if (sentence > num_lines) then
      break
    end
    assert(token <= num_tokens)
    local parts = Utils.split(line)
    assert(#parts > 0)
    if #parts <= config.kMaxSentenceSize then
      local size = Utils.ChunkSentence(parts, config.kSentenceSize)
      self.lengths_[sentence] = size
      self.starts_[sentence] = token
      for i = 1, size do
        assert(parts[i] ~= nil)
        self.tokens_[token] = word_vocab:word_id(parts[i])
        token = token + 1
      end
      sentence = sentence + 1
    end
  end
  assert(num_lines + 1 == sentence)
  assert(num_tokens + 1 == token)
  self.config_ = config
  if config.useGPU == true then
    require('cutorch')
  end
end

function DataSet:size()
  return self.lengths_:size(1)
end

