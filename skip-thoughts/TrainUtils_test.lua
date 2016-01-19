package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("rnn")
require("os")
require("TrainUtils")
require("Config")
require("Model")
require("VocabularyBuilder")
require("PolarityDataSet")

mytester = torch.Tester()

local TrainUtilsTest = {}

function TrainUtilsTest.Test()
  local config = Config()
  local vocab_builder = VocabularyBuilder(config)
  local word_vocab_file = paths.concat(config.kDataPath, "books.wrd.voc")
  local vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)
  local model = Model(config, vocab)
  local dataset= PolarityDataSet(config, vocab)
  local error_rate = TrainUtils.PolarityValidation(model, dataset)
  assert(error_rate ~= nil)
  assert(error_rate >= 0 and error_rate <= 1.0)
end

mytester:add(TrainUtilsTest)

mytester:run()


