require("torch")
require("paths")
require("Config")
require("Vocabulary")
require("ValidationDataSet")
require("Utils")

local PolarityDataSet, parent = torch.class("PolarityDataSet", 'ValidationDataSet')

function PolarityDataSet:__init(config, word_vocab)
  parent.__init(self)

  local positive_file = paths.concat(config.kDataPath, "rt.pos.wrd")
  local negative_file = paths.concat(config.kDataPath, "rt.neg.wrd")
  local positive_lines = Utils.ReadLines(positive_file)
  local negative_lines = Utils.ReadLines(negative_file)
  assert(positive_lines)
  assert(negative_lines)
  local positive_sample_lines = self:_Subrange(positive_lines, 1, config.kRTSampleSize)
  local negative_sample_lines = self:_Subrange(negative_lines, 1, config.kRTSampleSize)
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
  self.texts_ = {}
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

      local s = {}
      for i = 1, #parts do
        local id = word_vocab:word_id(parts[i])
        local text = word_vocab:word_text(id)
        self.tokens_[token] = id
        table.insert(s, text)
        token = token + 1
      end

      self.texts_[sentence] = table.concat(s, " ")
      self.sentences_[sentence] = line

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

function PolarityDataSet:Examples(nearest)
  local e = {}
  local p = 0
  local n = 0
  local size = self.lengths_:size(1)
  for i = 1, size do
    if self.labels_[i] == self.labels_[nearest[i]] then
      if p < 3 then
        table.insert(e, {1, i, nearest[i]})
      end
      p = p + 1
    else
      if n < 3 then
        table.insert(e, {-1, i, nearest[i]})
      end
      n = n + 1
    end
  end
  return e
end

function PolarityDataSet:Precision(nearest)
  assert(nearest:size(1) == self.lengths_:size(1))
  local correct = 0
  local size = nearest:size(1)
  for i = 1, size do
    if self.labels_[i] == self.labels_[nearest[i]] then
      correct = correct + 1
    end
  end
  return correct / size
end

function PolarityDataSet:Show(examples)
  ValidationDataSet.Show(examples, self.lengths_:size(1), self.texts_, self.sentences_)
end

function PolarityDataSet:_Subrange(t, s, l)
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
