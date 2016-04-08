require('torch')
require('math')

require('DataSet')
require('Utils')

local BatchSampler = torch.class('BatchSampler')

function BatchSampler:__init(config, data_set, indices)
  assert(config ~= nil)
  assert(data_set ~= nil)
  assert(config.kBatchSize >= 1)
  assert(data_set.lengths_ ~= nil)
  assert(data_set.tokens_ ~= nil)
  assert(data_set.starts_ ~= nil)
  self.data_set_ = data_set
  self.lengths_ = data_set.lengths_
  self.tokens_ = data_set.tokens_
  self.starts_ = data_set.starts_
  self.config_ = config
  self.indices_ = indices or torch.randperm(data_set.lengths_:size(1))
  self.current_ = 1
end

function BatchSampler:_GetStarts()
  local batch_size = math.min(self.config_.kBatchSize, self.indices_:size(1))
  local starts = torch.LongTensor(batch_size)
  for i = 1, batch_size do
    if self.current_ > self.indices_:size(1) then
      self.current_ = 1
    end
    starts[i] = self.current_
    self.current_ = self.current_ + 1
  end
  return starts
end

function BatchSampler:_GetBatchFeature(starts, max_length)
  local idx = torch.LongTensor(starts:size(1))
  for i = 1, idx:size(1) do
    idx[i] = self.indices_[starts[i]]
  end
  return self.data_set_.features_:index(1, idx)
end

function BatchSampler:_GetBatchScore(starts, max_length)
  local scores = torch.FloatTensor(starts:size(1), 1)
  for i = 1, starts:size(1) do
    local p = self.indices_[starts[i]]
    local s = self.data_set_.scores_[p][1]
    assert(Utils.IsNaN(s) == false)
    scores[i][1] = s
  end
  return scores
end

function BatchSampler:_GetBatchSentence(starts, max_length)
  local batch_size = starts:size(1)
  local sentence = torch.IntTensor(batch_size, max_length):fill(VocabularyUtils.kPadWordId)
  for k = 1, max_length do
    for i = 1, batch_size do
      local p = self.indices_[starts[i]]
      assert(self.lengths_[p] > 0)
      assert(self.starts_[p] >= 1 and self.starts_[p] + self.lengths_[p] - 1 <= self.tokens_:size(1))
      local t
      t = self.starts_[p] + k - 1
      if k <= self.lengths_[p] then
        sentence[i][k] = self.tokens_[t]
      end
    end
  end
  return sentence
end

function BatchSampler:_GetMaxLength(starts)
  local max_size = 0
  for i = 1, starts:size(1) do
    local s = self.indices_[starts[i]]
    local p = self.lengths_[s]
    max_size = math.max(p, max_size)
  end
  return max_size
end

function BatchSampler:SampleBatch()
  assert(self.config_.kBatchSize > 0)
  local batch = {}
  local label = {}
  local starts = self:_GetStarts()
  assert(starts:size(1) <= self.config_.kBatchSize)
  local max_prev_size = self:_GetMaxLength(starts)
  local tokens = self:_GetBatchSentence(starts, max_prev_size)
  local features = self:_GetBatchFeature(starts, max_prev_size)
  local scores = self:_GetBatchScore(starts, max_prev_size)
  table.insert(batch, tokens)
  table.insert(batch, features)
  table.insert(batch, scores)
  return batch
end

function BatchSampler:size()
  return self.data_set_:size()
end
