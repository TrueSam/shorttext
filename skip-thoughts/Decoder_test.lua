package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("rnn")
require("os")
require("Decoder")
require("Config")

mytester = torch.Tester()

local DecoderTest = {}

do
  local MockVocabulary = torch.class('MockVocabulary')

  function MockVocabulary:__init()
  end

  function MockVocabulary:size()
    return 10
  end
end

function DecoderTest.Test()
  local config = Config()
  local vocab = MockVocabulary()
  local table = Decoder.create(config, vocab)
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
  assert(prev_output[1]:size(2) == vocab:size())
  assert(#next_output == #next_sent)
  assert(next_output[1]:dim() == 2)
  assert(next_output[1]:size(1) == 2)
  assert(next_output[1]:size(2) == vocab:size())
end

mytester:add(DecoderTest)

mytester:run()


