require("rnn")
require("nn")
require("torch")
require("nngraph")
require("Vocabulary")
require("ConditionTable")
require("Config")
require("CGRU")

local Decoder = torch.class("Decoder")

function Decoder.create(config, word_vocab, lookup)
  local model = nn.Sequential()
  model:add(ConditionTable(config))
  local seq = nn.Sequential()
  seq:add(CGRU(config.kWordDim, config.kHiddenDim, config.kHiddenDim))
  local w = nn.LinearNoBias(config.kHiddenDim, word_vocab:size())
  w:shared(lookup, 'weight', 'gradWeight')
  seq:add(w)
  seq:add(nn.LogSoftMax())
  model:add(nn.Sequencer(seq))
  return model
end
