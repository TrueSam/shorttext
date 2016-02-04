package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("rnn")
require("os")
require("BucketBatchSampler")
require("Config")
require("DataSet")
require("VocabularyBuilder")
require("VocabularyUtils")

mytester = torch.Tester()

local BucketBatchSamplerTest = {}

function BucketBatchSamplerTest.Test()
  local config = Config()
  config.useGPU = false
  config.kMinWordFreq = 1
  config.kSampleSize = 7
  config.kBatchSize = 3
  config.kNumBucket = 3
  local vocab_builder = VocabularyBuilder(config)

  local word_vocab_file = paths.concat(config.kDataPath, "books.wrd.voc")
  local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)

  local word_file = paths.concat(config.kDataPath, "books.wrd")
  local dataset = DataSet(config, word_file, word_vocab)
  local sampler = BucketBatchSampler(config, dataset.tokens_, dataset.starts_, dataset.lengths_);

  local past_current = {}
  for s = 1, 100 do
    local batch, label = sampler:SampleBatch()
    table.insert(past_current, sampler.current_)
  end

  local identical = true
  for i = 1, #past_current - 1 do
    if past_current[i] ~= past_current[i + 1] then
      identical = false
    end
  end
  assert(identical == false)
end

mytester:add(BucketBatchSamplerTest)

mytester:run()


