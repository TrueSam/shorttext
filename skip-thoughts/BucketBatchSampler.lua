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
  buckets = Utils.MergeBuckets(buckets, config.kBatchSize)
  assert(#buckets > 0)
  local num_elements = 0
  for i = 1, #buckets do
    assert(buckets[i]:size(1) > 0)
    num_elements = num_elements + buckets[i]:size(1)
    for j = 1, buckets[i]:size(1) do
      assert(buckets[i][j] > 0 and buckets[i][j] <= lengths:size(1) - 2)
    end
  end
  assert(num_elements == lengths:size(1) - 2)
  self.indices_ = torch.IntTensor(num_elements)
  local start = 1
  for i = 1, #buckets do
    self.indices_:narrow(1, start, buckets[i]:size(1)):copy(buckets[i])
    start = start + buckets[i]:size(1)
  end
  assert(start == num_elements + 1)
  self.current_ = 1
end
