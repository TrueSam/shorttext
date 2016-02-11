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

function DataSetTest.BucketTest()
  local config = Config()
  config.kMinWordFreq = 1
  config.kSampleSize = 7
  config.kBatchSize = 3
  config.kNumBucket = 3
  config.kMinSentenceSize = 0
  config.kMaxSentenceSize = 100
  local vocab_builder = VocabularyBuilder(config)

  local word_vocab_file = paths.concat(config.kDataPath, "books.wrd.voc")
  local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)

  local word_file = paths.concat(config.kDataPath, "books.wrd")
  local dataset = DataSet(config, word_file, word_vocab)
  assert(dataset.lengths_:size(1) == config.kSampleSize)
  assert(dataset.lengths_:size(1) == config.kSampleSize)
  assert(dataset.starts_:size(1) == config.kSampleSize)
  assert(dataset.tokens_:size(1) == torch.sum(dataset.lengths_))

  assert(dataset.lengths_[1] == 13)  -- 3
  assert(dataset.lengths_[2] == 11)  -- 2
  assert(dataset.lengths_[3] == 12)  -- 2
  assert(dataset.lengths_[4] == 35)  -- 3
  assert(dataset.lengths_[5] == 25)  -- 3
  assert(dataset.lengths_[6] == 7)   -- 1
  assert(dataset.lengths_[7] == 2)   -- 1

  assert(dataset.starts_[1] == 1)
  assert(dataset.starts_[2] == 14)
  assert(dataset.starts_[3] == 25)
  assert(dataset.starts_[4] == 37)
  assert(dataset.starts_[5] == 72)
  assert(dataset.starts_[6] == 97)
  assert(dataset.starts_[7] == 104)
end

function DataSetTest.BatchTest()
  local config = Config()
  config.kMinWordFreq = 1
  config.kSampleSize = 7
  config.kBatchSize = 3
  config.kNumBucket = 1
  config.kMinSentenceSize = 0
  config.kMaxSentenceSize = 100
  local vocab_builder = VocabularyBuilder(config)

  local word_vocab_file = paths.concat(config.kDataPath, "books.wrd.voc")
  local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)

  local word_file = paths.concat(config.kDataPath, "books.wrd")
  local dataset = DataSet(config, word_file, word_vocab)
  assert(dataset.lengths_:size(1) == config.kSampleSize)
  assert(dataset.lengths_:size(1) == config.kSampleSize)
  assert(dataset.starts_:size(1) == config.kSampleSize)
  assert(dataset.tokens_:size(1) == torch.sum(dataset.lengths_))

  assert(dataset.lengths_[1] == 13)
  assert(dataset.lengths_[2] == 11)
  assert(dataset.lengths_[3] == 12)
  assert(dataset.lengths_[4] == 35)
  assert(dataset.lengths_[5] == 25)
  assert(dataset.lengths_[6] == 7)
  assert(dataset.lengths_[7] == 2)

  assert(dataset.starts_[1] == 1)
  assert(dataset.starts_[2] == 14)
  assert(dataset.starts_[3] == 25)
  assert(dataset.starts_[4] == 37)
  assert(dataset.starts_[5] == 72)
  assert(dataset.starts_[6] == 97)
  assert(dataset.starts_[7] == 104)

  local sampler = BatchSampler(config, dataset.tokens_, dataset.starts_, dataset.lengths_);

  for s = 1, 100 do
    local start = nil
    if sampler.current_ <= sampler.indices_:size(1) then
      start = sampler.indices_[sampler.current_]
    end
    local batch, label = sampler:SampleBatch()
    assert(#batch == 3)  -- prev, curr, next
    for k = 1, #batch do
      for i = 1, #batch[k] do
        assert(batch[k][i]:size(1), config.kBatchSize)
      end
    end
    if start == 1 then
      assert(batch[1][1][1], word_vocab:word_id('usually'))
      assert(batch[1][5][1], word_vocab:word_id('tearing'))
      assert(batch[1][10][1], word_vocab:word_id('playing'))
      assert(batch[1][12][1], word_vocab:word_id('his'))

      assert(batch[1][1][2], word_vocab:word_id('but'))
      assert(batch[1][5][2], word_vocab:word_id('at'))
      assert(batch[1][10][2], word_vocab:word_id('practically'))
      assert(batch[1][12][2], VocabularyUtils.kEndSentenceWordId)

      assert(batch[1][1][3], word_vocab:word_id('that'))
      assert(batch[1][5][3], word_vocab:word_id('\'s'))
      assert(batch[1][10][3], word_vocab:word_id('him'))
      assert(batch[1][12][3], word_vocab:word_id('earlier'))

      assert(batch[3][1][1], word_vocab:word_id('that'))
      assert(batch[3][5][1], word_vocab:word_id('\'s'))
      assert(batch[3][10][1], word_vocab:word_id('him'))
      assert(batch[3][12][1], word_vocab:word_id('earlier'))

      assert(batch[3][1][2], word_vocab:word_id('he'))
      assert(batch[3][5][2], word_vocab:word_id('movie'))
      assert(batch[3][10][2], word_vocab:word_id('he'))
      assert(batch[3][12][2], word_vocab:word_id('a'))

      assert(batch[3][1][3], word_vocab:word_id('she'))
      assert(batch[3][5][3], word_vocab:word_id('being'))
      assert(batch[3][10][3], word_vocab:word_id('older'))
      assert(batch[3][12][3], word_vocab:word_id('was'))
      elseif start == 3 then
        assert(batch[1][1][1], word_vocab:word_id('that'))
        assert(batch[1][5][1], word_vocab:word_id('\'s'))
        assert(batch[1][10][1], word_vocab:word_id('him'))
        assert(batch[1][12][1], word_vocab:word_id('earlier'))

        assert(batch[1][1][2], word_vocab:word_id('he'))
        assert(batch[1][5][2], word_vocab:word_id('movie'))
        assert(batch[1][10][2], word_vocab:word_id('he'))
        assert(batch[1][12][2], word_vocab:word_id('a'))

        assert(batch[1][1][3], word_vocab:word_id('she'))
        assert(batch[1][5][3], word_vocab:word_id('being'))
        assert(batch[1][10][3], word_vocab:word_id('older'))
        assert(batch[1][12][3], word_vocab:word_id('was'))

        assert(batch[3][1][1], word_vocab:word_id('she'))
        assert(batch[3][5][1], word_vocab:word_id('being'))
        assert(batch[3][10][1], word_vocab:word_id('older'))
        assert(batch[3][12][1], word_vocab:word_id('was'))

        assert(batch[3][1][2], word_vocab:word_id('are'))
        assert(batch[3][5][2], word_vocab:word_id('a'))
        assert(batch[3][10][2], VocabularyUtils.kEndSentenceWordId)
        assert(batch[3][12][2], VocabularyUtils.kEndSentenceWordId)

        assert(batch[3][1][3], word_vocab:word_id('she'))
        assert(batch[3][5][3], VocabularyUtils.kEndSentenceWordId)
        assert(batch[3][10][3], VocabularyUtils.kEndSentenceWordId)
        assert(batch[3][12][3], VocabularyUtils.kEndSentenceWordId)
      end
    end
end

function DataSetTest.Test()
  local config = Config()
  config.kMinWordFreq = 1
  config.kSentenceSize = -1
  config.kNumBucket = 1
  config.kSampleSize = 5
  config.kBatchSize = 1
  config.kMinSentenceSize = 0
  config.kMaxSentenceSize = 100
  local vocab_builder = VocabularyBuilder(config)

  local word_vocab_file = paths.concat(config.kDataPath, "books.wrd.voc")
  local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)

  local word_file = paths.concat(config.kDataPath, "books.wrd")
  local dataset = DataSet(config, word_file, word_vocab)
  assert(dataset.lengths_:size(1) == config.kSampleSize)
  assert(dataset.tokens_:size(1) == torch.sum(dataset.lengths_))
  assert(dataset.starts_:size(1) == config.kSampleSize)

  assert(dataset.lengths_[1] == 13)
  assert(dataset.lengths_[2] == 11)
  assert(dataset.lengths_[3] == 12)
  assert(dataset.lengths_[4] == 35)
  assert(dataset.lengths_[5] == 25)

  assert(dataset.starts_[1] == 1)
  assert(dataset.starts_[2] == 14)
  assert(dataset.starts_[3] == 25)
  assert(dataset.starts_[4] == 37)
  assert(dataset.starts_[5] == 72)

  assert(dataset.lengths_:size(1) == config.kSampleSize)

  local sampler = BatchSampler(config, dataset.tokens_, dataset.starts_, dataset.lengths_);

  for s = 1, 100 do
    local start = nil
    if sampler.current_ <= sampler.indices_:size(1) then
      start = sampler.indices_[sampler.current_]
    end
    local batch = sampler:SampleBatch()
    assert(#batch == 3)  -- prev, curr, next
    for k = 1, #batch do
      assert(#batch[k], dataset.lengths_[k])
      for i = 1, #batch[k] do
        assert(batch[k][i]:size(1), config.kBatchSize)
      end
    end
    if start == 1 then
      assert(#batch[1] == 13)
      assert(batch[1][1][1], word_vocab:word_id('usually'))
      assert(batch[1][5][1], word_vocab:word_id('tearing'))
      assert(batch[1][10][1], word_vocab:word_id('playing'))
      assert(batch[1][13][1], word_vocab:word_id('toys'))

      assert(#batch[2] == 11)
      assert(batch[2][1][1], word_vocab:word_id('but'))
      assert(batch[2][5][1], word_vocab:word_id('at'))
      assert(batch[2][10][1], word_vocab:word_id('practically'))
      assert(batch[2][11][1], word_vocab:word_id('catatonic'))

      assert(#batch[3] == 12)
      assert(batch[3][1][1], word_vocab:word_id('that'))
      assert(batch[3][5][1], word_vocab:word_id('\'s'))
      assert(batch[3][10][1], word_vocab:word_id('dressed'))
      assert(batch[3][12][1], word_vocab:word_id('catatonic'))

    elseif start == 3 then
      assert(#batch[1] == 12)
      assert(batch[1][1][1], word_vocab:word_id('that'))
      assert(batch[1][5][1], word_vocab:word_id('\'s'))
      assert(batch[1][10][1], word_vocab:word_id('dressed'))
      assert(batch[1][12][1], word_vocab:word_id('catatonic'))

      assert(#batch[2] == 35)
      assert(batch[2][1][1], word_vocab:word_id('he'))
      assert(batch[2][5][1], word_vocab:word_id('movie'))
      assert(batch[2][10][1], word_vocab:word_id('he'))
      assert(batch[2][35][1], word_vocab:word_id('older'))

      assert(#batch[3] == 25)
      assert(batch[3][1][1], word_vocab:word_id('she'))
      assert(batch[3][5][1], word_vocab:word_id('being'))
      assert(batch[3][10][1], word_vocab:word_id('older'))
      assert(batch[3][25][1], word_vocab:word_id('age'))
    end
  end

end

mytester:add(DataSetTest)

mytester:run()


