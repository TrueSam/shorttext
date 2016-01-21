require('torch')

local BatchSampler = torch.class('BatchSampler')

function BatchSampler:__init(config, tokens, starts, lengths)
  assert(config ~= nil)
  assert(lengths ~= nil)
  assert(config.kBatchSize > 0)
  self.lengths_ = lengths
  self.tokens_ = tokens
  self.starts_ = starts
  self.config_ = config
  -- The starts always from the first sentence.
  local max_index = lengths:size(1) - 2
  assert(max_index > 0)
  self.indices_ = torch.randperm(max_index)
  self.current_ = 1
end

function BatchSampler:_GetStarts()
  local batch_size = self.config_.kBatchSize
  local starts = torch.IntTensor(batch_size)
  for i = 1, batch_size do
    if self.current_ > self.indices_:size(1) then
      self.current_ = 1
    end
    local start = self.indices_[self.current_]
    starts[i] = start
  end
  return starts
end

function BatchSampler:_GetBatchSentence(batch_size, starts, max_length)
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

function BatchSampler:_GetMaxLength(starts)
  local max_size = 0
  for i = 1, starts:size(1) do
    local s = starts[i]
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
  assert(starts:size(1) == self.config_.kBatchSize)
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
  local prev_label = prev
  local next_label = next
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
