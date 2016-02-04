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

local function RandomBatch(vocab_size, batch_size, sentence_size)
  local sent = {}
  for i = 1, sentence_size do
    local s = torch.IntTensor(batch_size)
    for k = 1, batch_size do
      s[k] = torch.random(1, vocab_size)
    end
    table.insert(sent, s)
  end
  return sent
end

local function RandomTrainExample(vocab_size, batch_size, sentence_size)
  assert(vocab_size > 0)
  assert(sentence_size > 0)
  assert(batch_size > 0)
  local prev_size = torch.random(1, sentence_size)
  local curr_size = torch.random(1, sentence_size)
  local next_size = torch.random(1, sentence_size)
  local batch = torch.random(1, batch_size)
  local prev_sent = RandomBatch(vocab_size, batch, prev_size)
  local curr_sent = RandomBatch(vocab_size, batch, curr_size)
  local next_sent = RandomBatch(vocab_size, batch, next_size)

  local input = {prev_sent, curr_sent, next_sent}
  local target = {prev_sent, next_sent}

  return input, target
end

function ModelTest.Test()
  local config = Config()
  config.useGPU = false
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

  for i = 1, 100 do
    print('Running ' .. i .. ' random example')
    input, target = RandomTrainExample(vocab:size(), 100, 100)
    loss = table:train(input, target)
    assert(loss ~= nil)
    collectgarbage()
  end

  local v = table:vector(torch.IntTensor{1, 2, 3, 4, 5})
  assert(v ~= nil)
  assert(v:dim() == 1)
  assert(v:size(1) == config.kHiddenDim)
end

mytester:add(ModelTest)

mytester:run()


