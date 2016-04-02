package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("rnn")
require("os")
require("BatchSampler")
require("Config")
require("DataSet")
require("VocabularyBuilder")
require("VocabularyUtils")

mytester = torch.Tester()

local BatchSamplerTest = {}

function BatchSamplerTest.Test()
  local config = Config()
  config.useGPU = false
  config.kMinWordFreq = 1
  config.kSampleSize = 7
  config.kBatchSize = 1
  config.kNumBucket = 3
  local vocab_builder = VocabularyBuilder(config)

  local word_vocab_file = paths.concat(config.kTrainingDataPath, "vocabulary.txt")
  local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)

  local word_file = paths.concat(config.kTrainingDataPath, "sentences.txt")
  local dataset = DataSet(config, word_file, word_vocab)
  local sampler = BatchSampler(config, dataset);
  assert(sampler.indices_:size(1) == sampler.lengths_:size(1) - config.kBatchSize)

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

mytester:add(BatchSamplerTest)

mytester:run()


