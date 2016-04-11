package.path = package.path .. ";./?.lua"

require("torch")
require("os")
require("sys")
require('pl.path')

require("Config")
require("DataSet")
require("Model")
require("BatchSampler")
require("Utils")
require("Summarizer")
require("VocabularyBuilder")
require("VocabularyUtils")

torch.setdefaulttensortype('torch.FloatTensor')

local config = Config()

print("min word frequency : " .. config.kMinWordFreq)
print("word dim : " .. config.kWordDim)
print("sentence feature dim : " .. config.kSentenceFeatureDim)
print("document feature dim : " .. config.kDocumentFeatureDim)
print("batch size : " .. config.kBatchSize)
print("sample size : " .. config.kSampleSize)
print("sentence length : " .. config.kSentenceSize)
print("use CUDA : " .. tostring(config.useGPU))

local vocab_builder = VocabularyBuilder(config)

local word_vocab_file = path.join(config.kDataPath, "build/vocabulary.txt")
local word_file = path.join(config.kDataPath, "build/sentences.txt")

print('Load vocabulary from ' .. word_vocab_file)
local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)
print('Loaded ' .. word_vocab:size() .. ' tokens')

print('Load books dataset.')
local train_dataset = DataSet(config, word_file, word_vocab)
print('Create sampler from dataset.')
local sampler = BatchSampler(config, train_dataset);

-- Epochs
local E = 5000

print('Initialize model')
local model = Model(config, word_vocab)
model.model_:training()

collectgarbage()

local start = sys.clock()

model:train(sampler, E)

print(string.format("Train completed after %f seconds.", sys.clock() - start))

local filename = path.join(config.kDataPath, 'build', string.format("model-%d.t7", E))
print('Saving to file: ' .. filename)
torch.save(filename, model)
