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
  local word_vocab_file = paths.concat(config.kTrainingDataPath, "vocabulary.txt")
  local freqs = {3, 5, 10}
  for _, f in ipairs(freqs) do
    config.kMinWordFreq = f
    local vocab_builder = VocabularyBuilder(config)
    local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)
    print('Loaded ' .. word_vocab:size() .. ' tokens ' .. 'at frequency ' .. config.kMinWordFreq)
    assert(word_vocab:word_text(VocabularyUtils.kEndSentenceWordId) ==
    VocabularyUtils.kEndSentenceWordText)
    assert(word_vocab:word_id(VocabularyUtils.kEndSentenceWordText) ==
    VocabularyUtils.kEndSentenceWordId)
    assert(word_vocab:word_text(VocabularyUtils.kUnknownWordId) ==
    VocabularyUtils.kUnknownWordText)
    assert(word_vocab:word_id(VocabularyUtils.kUnknownWordText) ==
    VocabularyUtils.kUnknownWordId)
    local text = 'the'
    local id = word_vocab:word_id(text)
    local expected = word_vocab:word_text(id)
    print(expected, id)
    assert(expected == text)
  end
end

mytester:add(VocabularyBuilderTest)

mytester:run()


