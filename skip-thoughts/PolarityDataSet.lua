require("torch")
require("paths")
require("Config")
require("Vocabulary")
require("Utils")

local PolarityDataSet = torch.class("PolarityDataSet")

function PolarityDataSet:__init(config, word_vocab)
  local positive_file = paths.concat(config.kDataPath, "rt.pos.wrd")
  local negative_file = paths.concat(config.kDataPath, "rt.neg.wrd")
  local positive_lines = Utils.ReadLines(positive_file)
  local negative_lines = Utils.ReadLines(negative_file)
  assert(positive_lines)
  assert(negative_lines)
  local positive_sample_lines = self:_subrange(positive_lines, 1, config.kRTSampleSize)
  local negative_sample_lines = self:_subrange(negative_lines, 1, config.kRTSampleSize)
  local num_positive_tokens = Utils.CountTokens(positive_sample_lines)
  local num_negative_tokens = Utils.CountTokens(negative_sample_lines)
  local num_tokens = num_positive_tokens + num_negative_tokens
  local num_lines = #positive_sample_lines + #negative_sample_lines
  assert(num_lines > 0)
  assert(num_tokens > 0)
  self.tokens_ = torch.IntTensor(num_tokens)
  self.starts_ = torch.IntTensor(num_lines)
  self.lengths_ = torch.IntTensor(num_lines)
  self.labels_ = torch.IntTensor(num_lines)
  self.sentences_ = {}

  local token = 1
  local sentence = 1

  local lines = {positive_sample_lines, negative_sample_lines}
  local labels = {1, -1}
  for k = 1, #lines do
    local file = lines[k]
    local label = labels[k]
    for i = 1, #lines[k] do
      local line = lines[k][i]
      if sentence > num_lines then
        break
      end
      assert(token <= num_tokens)
      local parts = Utils.split(line)
      assert(#parts > 0)
      table.insert(self.sentences_, line)
      self.lengths_[sentence] = #parts
      self.starts_[sentence] = token
      self.labels_[sentence] = label
      for i = 1, #parts do
        self.tokens_[token] = word_vocab:word_id(parts[i])
        token = token + 1
      end
      sentence = sentence + 1
    end
  end

  assert(num_lines + 1 == sentence)
  assert(num_tokens + 1 == token)
  assert(num_lines == #self.sentences_)

  self.current_ = 1
  self.config_ = config
  self.size_ = num_lines
  self.word_vocab_ = word_vocab
  if config.useGPU == true then
    require('cutorch')
  end
end

function PolarityDataSet:_token_text(tokens)
  local s = {}
  for i = 1, tokens:size(1) do
    local t = tokens[i]
    table.insert(s, self.word_vocab_:word_text(t))
  end
  return table.concat(s, " ")
end

function PolarityDataSet:GetSentence(p)
  assert(p >= 1 and p <= self.size_)
  local tokens = self.tokens_:narrow(1, self.starts_[p], self.lengths_[p])
  local text = self:_token_text(tokens)
  if self.config_.useGPU == true then
    tokens = tokens:cuda()
  end
  local label = self.labels_[p]
  local original_text = self.sentences_[p]
  return tokens, label, text, original_text
end

function PolarityDataSet:size()
  return self.size_
end

function PolarityDataSet:_subrange(t, s, l)
  assert(type(t) == 'table')
  local r = {}
  for i = 0, l - 1 do
    local c = s + i
    if c > #t then
      break
    end
    table.insert(r, t[c])
  end
  return r
end
