package.path = package.path .. ";./?.lua"

require("torch")
require("logging")
require("logging.console")
require("os")
require("paths")
require("sys")

require("Config")
require("DataSet")
require("Model")
require("PolarityDataSet")
require("ManualDataSet")
require("Validator")
require("BatchSampler")
require("BucketBatchSampler")
require("Utils")
require("ManualDataSet")
require("VocabularyBuilder")

torch.setdefaulttensortype('torch.FloatTensor')

local logger = logging.console()
local config = Config()

logger:info("min word frequency : " .. config.kMinWordFreq)
logger:info("word dim : " .. config.kWordDim)
logger:info("hidden dim : " .. config.kHiddenDim)
logger:info("batch size : " .. config.kBatchSize)
logger:info("sample size : " .. config.kSampleSize)
logger:info("sentence length : " .. config.kSentenceSize)
logger:info("use CUDA : " .. tostring(config.useGPU))

local vocab_builder = VocabularyBuilder(config)

local word_vocab_file = paths.concat(config.kDataPath, "books.wrd.voc")
local word_file = paths.concat(config.kDataPath, "books.wrd")

print('Load vocabulary from ' .. word_vocab_file)
local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)
print('Loaded ' .. word_vocab:size() .. ' tokens')

print('Load books dataset.')
local train_dataset = DataSet(config, word_file, word_vocab)
print('Create sampler from dataset.')
-- local sampler = BatchSampler(config, train_dataset.tokens_, train_dataset.starts_, train_dataset.lengths_);
local sampler = BucketBatchSampler(config, train_dataset.tokens_, train_dataset.starts_, train_dataset.lengths_);

print('Load rt polarity dataset.')
local rt_dataset = PolarityDataSet(config, word_vocab)
local rt_validator = Validator(config, rt_dataset.tokens_, rt_dataset.starts_, rt_dataset.lengths_)
print('Load manual dataset.')
local manual_dataset = ManualDataSet(config, word_vocab)
local manual_validator = Validator(config, manual_dataset.tokens_, manual_dataset.starts_, manual_dataset.lengths_)

-- Epochs
local E = 5

print('Initialize model')
local model = Model(config, word_vocab)
model.model_:training()

collectgarbage()

local start = sys.clock()
for e = 1, E do
  collectgarbage()
  local size = train_dataset:size()
  assert(size > 0)

  -- validation after very M samples
  local M = 20
  for i = 1, size do
    local sentences, targets = sampler:SampleBatch()
    assert(sentences ~= nil)
    assert(targets ~= nil)
    local loss = model:train(sentences, targets)
    assert(loss ~= nil)

    if i % M == 0 then
      local nearest = rt_validator:FindNearest(model, Validator.COSINE)
      local examples = rt_dataset:Examples(nearest)
      local error_rate = 1 - rt_dataset:Precision(nearest)
      logger:info(string.format("Error rate for tag prediction on rt dev set using cosine: %f", error_rate))
      -- rt_dataset:show(examples)

      nearest = rt_validator:FindNearest(model, Validator.EUCLIDEAN)
      examples = rt_dataset:Examples(nearest)
      error_rate = 1 - rt_dataset:Precision(nearest)
      logger:info(string.format("Error rate for tag prediction on rt dev set using euclidean: %f", error_rate))
      -- rt_dataset:show(examples)

      nearest = manual_validator:FindNearest(model, Validator.COSINE)
      examples = manual_dataset:Examples(nearest)
      error_rate = 1 - manual_dataset:Precision(nearest)
      logger:info(string.format("Error rate for tag prediction on manual dev set using cosine: %f", error_rate))
      -- manual_dataset:show(examples)

      nearest = manual_validator:FindNearest(model, Validator.EUCLIDEAN)
      examples = manual_dataset:Examples(nearest)
      error_rate = 1 - manual_dataset:Precision(nearest)
      logger:info(string.format("Error rate for tag prediction on manual dev set using euclidean: %f", error_rate))
      -- manual_dataset:show(examples)
    end

    if i % 100 == 0 then
      logger:info("Epoch " .. e .. " trained " .. i .. " examples, " .. 'current loss: ' .. loss)
      collectgarbage()
    end
  end
end

logger:info(string.format("Train completed after %f seconds.", sys.clock() - start))

local filename = paths.concat(string.format("model-%d.t7", E))
torch.save(filename, model:float())
