package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("rnn")
require("os")
require("PolarityDataSet")
require("Config")
require("Vocabulary")
require("VocabularyBuilder")

mytester = torch.Tester()

local PolarityDataSetTest = {}

function PolarityDataSetTest.Test()
  local config = Config()
local vocab_builder = VocabularyBuilder(config)
  local word_vocab_file = paths.concat(config.kDataPath, "books.wrd.voc")
  local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)
  local table = PolarityDataSet(config, word_vocab)
  assert(table:size() > 1)
  local s = table:GetSentence(1)
  assert(s:size(1) > 0)
end

mytester:add(PolarityDataSetTest)

mytester:run()


