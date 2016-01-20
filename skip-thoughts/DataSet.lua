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
  self.limits_ = Utils.BuildBuckets(self.lengths_, config.kNumBucket)
  self.buckets_ = Utils.DistributeBuckets(self.lengths_, self.limits_)
  local non_empty_buckets = {}
  for i = 1, #self.buckets_ do
    if #self.buckets_[i] > 0 then
      table.insert(non_empty_buckets, i)
    end
  end
  self.bucket_index_ = Utils.RandomPermute(non_empty_buckets)
  self.indices_ = torch.IntTensor(#self.bucket_index_):fill(1)
  self.current_ = 1
  self.config_ = config
  if config.useGPU == true then
    require('cutorch')
  end
end

function DataSet:_GetBatchLabel(batch_size, starts, max_length)
  local label = {}
  assert(batch_size == starts:size(1))
  for k = 1, max_length do
    local pl = torch.IntTensor(batch_size):fill(VocabularyUtils.kEndSentenceWordId)

    for i = 1, batch_size do
      local p = starts[i]
      assert(self.lengths_[p] > 0)
      assert(self.starts_[p] >= 1 and self.starts_[p] + self.lengths_[p] - 1 <= self.tokens_:size(1))
      local t
      t = self.starts_[p] + k - 1
      if k <= self.lengths_[p] then
        pl[i] = self.tokens_[t]
      end
    end
    table.insert(label, pl)
  end
  return label
end

function DataSet:_GetBatchSentence(batch_size, starts, max_length)
  local sentence = {}
  assert(batch_size == starts:size(1))
  for k = 1, max_length do
    local pt = torch.IntTensor(batch_size):fill(VocabularyUtils.kEndSentenceWordId)

    for i = 1, batch_size do
      local p = starts[i]
      assert(self.lengths_[p] > 0)
      assert(self.starts_[p] >= 1 and self.starts_[p] + self.lengths_[p] - 1 <= self.tokens_:size(1))
      local t
      t = self.starts_[p] + k - 1
      if k <= self.lengths_[p] then
        pt[i] = self.tokens_[t]
      end
    end
    table.insert(sentence, pt)
  end
  return sentence
end

function DataSet:_GetStarts()
  if self.current_ > #self.bucket_index_ then
    self.current_ = 1
  end
  local bucket_index = self.bucket_index_[self.current_]
  local current_bucket = self.buckets_[bucket_index]
  assert(#current_bucket > 0)
  local starts = torch.IntTensor(self.config_.kBatchSize)
  local i = 1
  while i <= self.config_.kBatchSize do
    if self.indices_[bucket_index] > #current_bucket then
      self.indices_[bucket_index] = 1
    end
    local start = current_bucket[self.indices_[bucket_index]]
    if start + 2 <= self.lengths_:size(1) then
      starts[i] = start
      i = i + 1
    end
  end
  self.current_ = self.current_ + 1
  return starts
end

function DataSet:_GetMaxLength(starts)
  local max_size = 0
  for i = 1, starts:size(1) do
    local s = starts[i]
    local p = self.lengths_[s]
    max_size = math.max(p, max_size)
  end
  return max_size
end

function DataSet:SampleBatch()
  assert(self.config_.kBatchSize > 0)
  local batch = {}
  local label = {}
  local starts = self:_GetStarts()
  local max_prev_size = self:_GetMaxLength(starts)
  local max_curr_size = self:_GetMaxLength(starts + 1)
  local max_next_size = self:_GetMaxLength(starts + 2)
  assert(max_prev_size > 0)
  assert(max_curr_size > 0)
  assert(max_next_size > 0)
  assert(self.config_.kBatchSize > 0)
  local prev = self:_GetBatchSentence(self.config_.kBatchSize, starts, max_prev_size)
  local curr = self:_GetBatchSentence(self.config_.kBatchSize, starts + 1, max_curr_size)
  local next = self:_GetBatchSentence(self.config_.kBatchSize, starts + 2, max_next_size)
  local prev_label = self:_GetBatchLabel(self.config_.kBatchSize, starts, max_prev_size)
  local next_label = self:_GetBatchLabel(self.config_.kBatchSize, starts + 2, max_next_size)

  table.insert(batch, prev)
  table.insert(batch, curr)
  table.insert(batch, next)

  table.insert(label, prev_label)
  table.insert(label, next_label)

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
  return self.lengths_:size(1)
end

