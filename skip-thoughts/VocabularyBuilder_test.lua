package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("rnn")
require("os")
require("VocabularyBuilder")
require("Config")

mytester = torch.Tester()

local VocabularyBuilderTest = {}

function VocabularyBuilderTest.Test()
  local config = Config()
  local word_vocab_file = paths.concat(config.kDataPath, "books.wrd.voc")
  local freqs = {3, 5, 10, 20, 50, 100}
  for _, f in ipairs(freqs) do
    config.kMinWordFreq = f
    local vocab_builder = VocabularyBuilder(config)
    local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)
    print('Loaded ' .. word_vocab:size() .. ' tokens ' .. 'at frequency ' .. config.kMinWordFreq)
  end
end

mytester:add(VocabularyBuilderTest)

mytester:run()

