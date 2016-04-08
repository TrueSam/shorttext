require('torch')
require('BatchSampler')
require('DataSet')
require('Utils')

local BucketBatchSampler, parent = torch.class('BucketBatchSampler')

BucketBatchSampler.kDefaultBuckets = {
  {1, 5},
  {6, 10},
  {11, 15},
  {16, 20},
  {21, 25},
  {26, 30},
  {31, 35},
  {36, 40},
  {41, 45},
  {46, 50},
  {51, 55},
  {56, 60},
  {61, 65},
  {66, 70},
  {71, 75},
  {76, 80},
  {81, 85},
  {86, 90},
  {91, 95},
  {96, 100},
  {101, 1000000}
}

function BucketBatchSampler:__init(config, dataset, buckets)
  assert(config ~= nil)
  assert(dataset ~= nil)
  buckets = buckets or BucketBatchSampler.kDefaultBuckets
  assert(#buckets > 0)
  for i = 1, #buckets do
    assert(#buckets[i] == 2)
    assert(buckets[i][1] <= buckets[i][2])
    if i < #buckets then
      assert(buckets[i][2] + 1 == buckets[i + 1][1])
    end
  end
  local bucket_count = {}
  for i = 1, #buckets do
    bucket_count[i] = {}
  end
  for i = 1, dataset.lengths_:size(1) do
    local b = -1
    for j = 1, #buckets do
      if buckets[j][2] >= dataset.lengths_[i] and buckets[j][1] <= dataset.lengths_[i] then
        b = j;
        break;
      end
    end
    assert(b >= 1 and b <= #buckets)
    table.insert(bucket_count[b], i);
  end

  local bucket_indices = {}
  for i = 1, #bucket_count do
    if #bucket_count[i] > 0 then
      table.insert(bucket_indices, torch.IntTensor(Utils.RandomPermute(bucket_count[i])))
    end
  end
  self.samplers_ = {}
  for i = 1, #bucket_indices do
    table.insert(self.samplers_, BatchSampler(config, dataset, bucket_indices[i]))
  end

  self.data_set_ = data_set
end

function BucketBatchSampler:SampleBatch()
  local b = torch.random(1, #self.samplers_)
  return self.samplers_[b]:SampleBatch()
end

function BucketBatchSampler:size()
  return self.data_set_:size()
end
