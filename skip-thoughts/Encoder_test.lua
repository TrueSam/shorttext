package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("os")
require("rnn")
require("Config")
require("DataSet")
require("Encoder")
require("VocabularyBuilder")
require("VocabularyUtils")

mytester = torch.Tester()

local EncoderTest = {}

function EncoderTest.BatchTest()
  local config = Config()

  local batch = {torch.LongTensor{1, 2}, torch.LongTensor{2, 3}}

  local lookup = nn.LookupTable(500000, config.kWordDim)
  local table = Encoder.createSequence(config, lookup)
  local output = table:forward(batch)
  assert(output ~= nil)
  assert(#output == 2)
  assert(output[1]:size(1) == 2)
  assert(output[1]:size(2) == config.kWordDim)
  assert(output[2]:size(1) == 2)
  assert(output[2]:size(2) == config.kWordDim)

  table = Encoder.create(config, lookup)
  output = table:forward(batch)
  assert(output ~= nil)
  assert(output:size(1) == 2)
  assert(output:size(2) == config.kHiddenDim)
end

mytester:add(EncoderTest)

mytester:run()


