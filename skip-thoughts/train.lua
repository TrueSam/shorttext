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
require("TrainUtils")
require("Utils")
require("VectorDataSet")
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

print('Load rt polarity dataset.')
local rt_dataset = PolarityDataSet(config, word_vocab)

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
  local M = 2000
  for i = 1, size do
    local sentences, targets = train_dataset:SampleBatch()
    local loss = model:train(sentences, targets)
    assert(loss ~= nil)

    if i % M == 0 then
      local error_rate = TrainUtils.PolarityValidation(model, rt_dataset, VectorDataSet.COSINE)
      logger:info(string.format("Error rate for tag prediction on dev set using cosine: %f", error_rate))
      error_rate = TrainUtils.PolarityValidation(model, rt_dataset, VectorDataSet.EUCLIDEAN)
      logger:info(string.format("Error rate for tag prediction on dev set using euclidean: %f", error_rate))
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
