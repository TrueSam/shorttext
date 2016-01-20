package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("rnn")
require("os")
require("DataSet")
require("Config")
require("VocabularyBuilder")
require("VocabularyUtils")

mytester = torch.Tester()

local DataSetTest = {}

function DataSetTest.BucketTest()
  local config = Config()
  config.kMinWordFreq = 1
  config.kSampleSize = 7
  config.kBatchSize = 3
  config.kNumBucket = 3
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

  assert(#dataset.limits_ == config.kNumBucket)
  assert(dataset.limits_[1] == 7)
  assert(dataset.limits_[2] == 12)
  assert(dataset.limits_[3] == 35)

  assert(dataset.buckets_[23][1] == 1)
  local past_current = {}
  for s = 1, 100 do
    local batch, label = dataset:SampleBatch()
    table.insert(past_current, dataset.current_)
  end

  local identical = true
  for i = 1, #past_current - 1 do
    if past_current[i] ~= past_current[i + 1] then
      identical = false
    end
  end
  assert(identical == false)
end

function DataSetTest.BatchTest()
  local config = Config()
  config.kMinWordFreq = 1
  config.kSampleSize = 7
  config.kBatchSize = 3
  config.kNumBucket = 1
  local vocab_builder = VocabularyBuilder(config)

  local word_vocab_file = paths.concat(config.kDataPath, "books.wrd.voc")
  local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)

  local word_file = paths.concat(config.kDataPath, "books.wrd")
  local dataset = DataSet(config, word_file, word_vocab)
  assert(dataset.lengths_:size(1) == config.kSampleSize)
  table.sort(dataset.buckets_[1])
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



  for s = 1, 100 do
    local start = dataset.indices_[1]
    local batch, label = dataset:SampleBatch()
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
  local vocab_builder = VocabularyBuilder(config)

  local word_vocab_file = paths.concat(config.kDataPath, "books.wrd.voc")
  local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)

  local word_file = paths.concat(config.kDataPath, "books.wrd")
  local dataset = DataSet(config, word_file, word_vocab)
  assert(dataset.lengths_:size(1) == config.kSampleSize)
  table.sort(dataset.buckets_[1])
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

  for s = 1, 100 do
    local start = dataset.indices_[1]
    local batch = dataset:SampleBatch()
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


