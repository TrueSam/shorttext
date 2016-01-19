-- The torch support large object only for Tensor, so we need to load the
-- sentences as tensor.
require("torch")
require("Utils")

local utf8 = require("lua-utf8")

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
  self.indices_ = torch.randperm(num_lines)
  self.current_ = 1
  self.config_ = config
  if config.useGPU == true then
    require('cutorch')
  end
end

function DataSet:SampleBatch()
  assert(self.config_.kBatchSize > 0)
  if self.config_.kSentenceSize <= 0 then
    assert(self.config_.kBatchSize == 1)
  end
  local batch = {}
  local label = {}
  local starts = torch.IntTensor(self.config_.kBatchSize)
  local i = 1
  while i <= self.config_.kBatchSize do
    if self.current_ + self.config_.kBatchSize > self.indices_:size(1) then
      self.current_ = 1
    end
    local start = self.indices_[self.current_]
    if start + 2 <= self.indices_:size(1) then
      starts[i] = start
      i = i + 1
    end
  end
  local prev = {}
  local curr = {}
  local next = {}
  local prev_label = {}
  local next_label = {}
  if self.config_.kBatchSize > 1 then
    assert(self.config_.kSentenceSize > 0)
    for k = 1, self.config_.kSentenceSize do
      local pt = torch.IntTensor(self.config_.kBatchSize):fill(VocabularyUtils.kEndSentenceWordId)
      local ct = torch.IntTensor(self.config_.kBatchSize):fill(VocabularyUtils.kEndSentenceWordId)
      local nt = torch.IntTensor(self.config_.kBatchSize):fill(VocabularyUtils.kEndSentenceWordId)

      local pl = torch.IntTensor(self.config_.kBatchSize):fill(VocabularyUtils.kEndSentenceWordId)
      local nl = torch.IntTensor(self.config_.kBatchSize):fill(VocabularyUtils.kEndSentenceWordId)

      for i = 1, self.config_.kBatchSize do
        local p = starts[i]
        local c = starts[i] + 1
        local n = starts[i] + 2
        assert(self.lengths_[p] > 0)
        assert(self.lengths_[c] > 0)
        assert(self.lengths_[n] > 0)
        assert(self.starts_[p] >= 1 and self.starts_[p] + self.lengths_[p] - 1 <= self.tokens_:size(1))
        assert(self.starts_[c] >= 1 and self.starts_[c] + self.lengths_[c] - 1 <= self.tokens_:size(1))
        assert(self.starts_[n] >= 1 and self.starts_[n] + self.lengths_[n] - 1 <= self.tokens_:size(1))
        local t
        t = self.starts_[p] + k - 1
        if k <= self.lengths_[p] then
          pt[i] = self.tokens_[t]
          pl[i] = self.tokens_[t]
        end
        t = self.starts_[c] + k - 1
        if k <= self.lengths_[c] then
          ct[i] = self.tokens_[t]
        end
        t = self.starts_[n] + k - 1
        if k <= self.lengths_[n] then
          nt[i] = self.tokens_[n]
          nl[i] = self.tokens_[n]
        end
      end
      table.insert(prev, pt)
      table.insert(curr, ct)
      table.insert(next, nt)

      table.insert(prev_label, pl)
      table.insert(next_label, nl)
    end
    table.insert(batch, prev)
    table.insert(batch, curr)
    table.insert(batch, next)

    table.insert(label, prev_label)
    table.insert(label, next_label)
  else
    for i = 1, self.config_.kBatchSize do
      local p = starts[i]
      local c = starts[i] + 1
      local n = starts[i] + 2
        assert(self.lengths_[p] > 0)
        assert(self.lengths_[c] > 0)
        assert(self.lengths_[n] > 0)
        assert(self.starts_[p] >= 1 and self.starts_[p] + self.lengths_[p] - 1 <= self.tokens_:size(1))
        assert(self.starts_[c] >= 1 and self.starts_[c] + self.lengths_[c] - 1 <= self.tokens_:size(1))
        assert(self.starts_[n] >= 1 and self.starts_[n] + self.lengths_[n] - 1 <= self.tokens_:size(1))

        local prev = {}
        local curr = {}
        local next = {}

        local prev_label = {}
        local next_label = {}
        for i = 1, self.lengths_[p] do
          local t = torch.IntTensor(1)
          t[1] = self.tokens_[self.starts_[p] + i - 1]
          table.insert(prev, t)
          table.insert(prev_label, t:clone())
        end
        for i = 1, self.lengths_[c] do
          local t = torch.IntTensor(1)
          t[1] = self.tokens_[self.starts_[c] + i - 1]
          table.insert(curr, t)
        end
        for i = 1, self.lengths_[n] do
          local t = torch.IntTensor(1)
          t[1] = self.tokens_[self.starts_[n] + i - 1]
          table.insert(next, t)
          table.insert(next_label, t:clone())
        end
        table.insert(batch, prev)
        table.insert(batch, curr)
        table.insert(batch, next)

        table.insert(label, prev_label)
        table.insert(label, next_label)
    end
  end
  if self.config_.useGPU == true then
    for i = 1, #batch do
      assert(#batch[i] > 0)
      for k = 1, #batch[i] do
        batch[i][k] = batch[i][k]:cuda()
      end
    end
    for i = 1, #label do
      assert(#label[i] > 0)
      for k = 1, #label[i] do
        label[i][k] = label[i][k]:cuda()
      end
    end
  end
  return batch, label
end

function DataSet:size()
  return self.indices_:size(1)
end

