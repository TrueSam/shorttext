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
  local vocab_builder = VocabularyBuilder(config)

  local word_vocab_file = paths.concat(config.kDataPath, "build/vocabulary.txt")
  local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)

  local word_file = paths.concat(config.kDataPath, "build/sentences.txt")
  local dataset = DataSet(config, word_file, word_vocab)
  local sampler = BucketBatchSampler(config, dataset);

  for s = 1, 100 do
    local batch = sampler:SampleBatch()
    print(batch)
  end
end

mytester:add(BucketBatchSamplerTest)

mytester:run()


