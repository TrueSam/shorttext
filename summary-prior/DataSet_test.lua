package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("rnn")
require("os")
require("DataSet")
require("Config")
require("VocabularyBuilder")
require("BatchSampler")
require("VocabularyUtils")

mytester = torch.Tester()

local DataSetTest = {}

function DataSetTest.Test()
  local config = Config()
  config.kMinWordFreq = 1
  config.kSampleSize = 3
  config.kBatchSize = 3
  config.kNumBucket = 3
  config.kMinSentenceSize = 0
  config.kMaxSentenceSize = 100
  local vocab_builder = VocabularyBuilder(config)

  local word_vocab_file = paths.concat(config.kTrainingDataPath, "vocabulary.txt")
  local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)

  local word_file = paths.concat(config.kTrainingDataPath, "sentences.txt")
  local dataset = DataSet(config, word_file, word_vocab)
  assert(dataset.lengths_:size(1) == config.kSampleSize)
  assert(dataset.lengths_:size(1) == config.kSampleSize)
  assert(dataset.starts_:size(1) == config.kSampleSize)
  assert(dataset.tokens_:size(1) == torch.sum(dataset.lengths_))

  assert(dataset.starts_[1] == 1)
  assert(dataset.tokens_[1] == word_vocab:word_id('washington'))
  assert(dataset.tokens_[8] == word_vocab:word_id('nation\'s'))
end

mytester:add(DataSetTest)

mytester:run()


