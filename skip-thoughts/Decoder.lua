require("rnn")
require("nn")
require("torch")
require("nngraph")
require("Vocabulary")
require("ConditionTable")
require("Config")
require("CGRU")

local Decoder = torch.class("Decoder")

function Decoder.create(config, word_vocab)
  local model = nn.Sequential()
  model:add(ConditionTable(config))
  local seq = nn.Sequential()
  seq:add(CGRU(config.kWordDim, config.kHiddenDim, config.kHiddenDim)):add(nn.Linear(config.kHiddenDim, word_vocab:size())):add(nn.LogSoftMax())
  model:add(nn.Sequencer(seq))
  return model
end
