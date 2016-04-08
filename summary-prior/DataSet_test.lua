package.path = package.path .. ";./?.lua"

require("torch")
require("pl.utils")
require("pl.test")
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

  local word_vocab_file = paths.concat(config.kDataPath, "build/vocabulary.txt")
  local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)
  local word_file = os.tmpname()
  local sentence = "beijing _ protesting the lack of a defense lawyer , the father of a prominent dissident is to seek a delay in his son's subversion trial , scheduled to start on thursday in the central city of wuhan .	POS 0	ROUGE-1 R:0.57576 P:0.25000 F:0.34862	ROUGE-2 R:0.17241 P:0.06944 F:0.09900"
  utils.writefile(word_file, sentence)

  local dataset = DataSet(config, word_file, word_vocab)
  assert(dataset.lengths_:size(1) == 1)
  assert(dataset.lengths_:size(1) == 1)
  assert(dataset.starts_:size(1) == 1)
  assert(dataset.tokens_:size(1) == torch.sum(dataset.lengths_))

  assert(dataset.starts_[1] == 1)
  assert(dataset.tokens_[1] == word_vocab:word_id('beijing'))
  assert(dataset.tokens_[8] == word_vocab:word_id('defense'))

  assert(dataset.scores_[1]:size(1) == 2)
  test.asserteq(dataset.scores_[1][1], 0.34862, 1e-4)
  test.asserteq(dataset.scores_[1][2], 0.09900, 1e-4)
end

mytester:add(DataSetTest)

mytester:run()


