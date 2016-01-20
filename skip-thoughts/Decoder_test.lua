package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("rnn")
require("os")
require("Decoder")
require("Config")
require("VocabularyBuilder")

mytester = torch.Tester()

local DecoderTest = {}

function DecoderTest.Test()
  local config = Config()
  config.kHiddenDim = config.kWordDim
  local vocab_builder = VocabularyBuilder(config)
  local word_vocab_file = paths.concat(config.kDataPath, "books.wrd.voc")
  local word_vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)
  local lookup = nn.LookupTable(word_vocab:size(), config.kWordDim)
  local table = Decoder.create(config, word_vocab, lookup)
  local prev_sent = {torch.randn(2, config.kWordDim), torch.randn(2, config.kWordDim), torch.randn(2, config.kWordDim)}
  local curr_sent = torch.randn(2, config.kHiddenDim)
  local next_sent = {torch.randn(2, config.kWordDim), torch.randn(2, config.kWordDim), torch.randn(2, config.kWordDim)}

  local prev_input = {prev_sent, curr_sent}
  local prev_output = table:forward(prev_input)

  local next_input = {next_sent, curr_sent}
  local next_output = table:forward(next_input)

  assert(#prev_output == #prev_sent)
  assert(prev_output[1]:dim() == 2)
  assert(prev_output[1]:size(1) == 2)
  assert(prev_output[1]:size(2) == word_vocab:size())
  assert(#next_output == #next_sent)
  assert(next_output[1]:dim() == 2)
  assert(next_output[1]:size(1) == 2)
  assert(next_output[1]:size(2) == word_vocab:size())
end

mytester:add(DecoderTest)

mytester:run()


