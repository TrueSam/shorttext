package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("os")
require("Model")
require("Config")
require("VocabularyBuilder")

mytester = torch.Tester()

local ModelTest = {}

do
  local MockVocabulary = torch.class('MockVocabulary')

  function MockVocabulary:__init()
  end

  function MockVocabulary:size()
    return 10
  end
end

function ModelTest.Test()
  local config = Config()
  local vocab_builder = VocabularyBuilder(config)
  local word_vocab_file = paths.concat(config.kDataPath, "books.wrd.voc")
  local vocab = vocab_builder:BuildWordVocabulary(word_vocab_file)
  local table = Model(config, vocab)
  local prev_sent = {torch.IntTensor{1, 2}, torch.IntTensor{2, 3}, torch.IntTensor{3, 4}, torch.IntTensor{4, 5}}
  local curr_sent = {torch.IntTensor{3, 4}, torch.IntTensor{2, 3}, torch.IntTensor{5, 6}, torch.IntTensor{7, 8}}
  local next_sent = {torch.IntTensor{4, 5}, torch.IntTensor{3, 4}, torch.IntTensor{8, 9}, torch.IntTensor{6, 7}}
  local input = {prev_sent, curr_sent, next_sent}
  local target = {prev_sent, next_sent}
  local output = table.model_:forward(input)
  assert(#output == 2)
  assert(#output[1] == 4)
  assert(output[1][1]:dim() == 2)
  assert(output[1][1]:size(1) == 2)
  assert(output[1][1]:size(2) == vocab:size())
  local loss = table.criterion_:forward(output, target)
  assert(loss ~= nil)
  local dp = table.criterion_:backward(output, target)
  assert(dp ~= nil)
  local t = table.model_:backward(input, dp)

  loss = table:train(input, target)
  assert(loss ~= nil)

  local v = table:vector(torch.IntTensor{1, 2, 3, 4, 5})
  assert(v ~= nil)
  assert(v:dim() == 1)
  assert(v:size(1) == config.kHiddenDim)
end

mytester:add(ModelTest)

mytester:run()


