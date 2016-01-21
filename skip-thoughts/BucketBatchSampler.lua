require('torch')
require('BatchSampler')
require('Utils')

local BucketBatchSampler, parent = torch.class('BucketBatchSampler', 'BatchSampler')

function BucketBatchSampler:__init(config, tokens, starts, lengths)
  parent.__init(self, config, tokens, starts, lengths)

  local limits = Utils.BuildBuckets(lengths, config.kNumBucket)
  assert(limits ~= nil)
  assert(#limits <= config.kNumBucket)
  local buckets = Utils.DistributeBuckets(lengths, limits)
  assert(buckets ~= nil)
  self.buckets_ = Utils.MergeBuckets(buckets, config.kBatchSize)
  assert(#self.buckets_ > 0)
  for i = 1, #self.buckets_ do
    assert(self.buckets_[i]:size(1) > 0)
    for j = 1, self.buckets_[i]:size(1) do
      assert(self.buckets_[i][j] > 0 and self.buckets_[i][j] <= lengths:size(1) - 2)
    end
  end
  self.bucket_indices_ = torch.randperm(#self.buckets_)
  self.indices_ = torch.IntTensor(#self.buckets_):fill(1)
  self.current_ = 1
end

function BucketBatchSampler:_GetStartsFromBucket(bucket_index)
  assert(bucket_index > 0 and bucket_index <= #self.buckets_)

  local current_bucket = self.buckets_[bucket_index]
  assert(current_bucket:size(1) > 0)

  local starts = torch.IntTensor(self.config_.kBatchSize)
  local index = self.indices_[bucket_index]
  assert(index ~= nil)
  assert(type(index) == 'number')
  for i = 1, self.config_.kBatchSize do
    if index > current_bucket:size(1) then
      index = 1
    end
    local start = current_bucket[index]
    assert(start ~= nil)
    assert(type(start) == 'number')
    assert(start > 0 and start <= self.lengths_:size(1) - 2)
    starts[i] = start
    index = index + 1
  end
  self.indices_[bucket_index] = index
  for i = 1, starts:size(1) do
    assert(starts[i] > 0 and starts[i] <= self.lengths_:size(1) - 2)
  end
  return starts
end

function BucketBatchSampler:_GetStarts()
  if self.current_ > self.bucket_indices_:size(1) then
    self.current_ = 1
  end
  local bucket_index = self.bucket_indices_[self.current_]
  local starts = self:_GetStartsFromBucket(bucket_index)
  self.current_ = self.current_ + 1
  return starts
end
