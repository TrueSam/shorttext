require("rnn")
require("nn")
require("torch")
require("nngraph")
require("Vocabulary")
require("ConditionTable")
require("LookupLinearNoBias")
require("Config")
require("CGRU")

local Decoder = torch.class("Decoder")

function Decoder.create(config, word_vocab, lookup)
  assert(config ~= nil)
  assert(word_vocab ~= nil)
  assert(lookup ~= nil)
  assert(config.kWordDim > 0)
  assert(config.kHiddenDim > 0)
  assert(word_vocab:size() > 0)
  local model = nn.Sequential()
  model:add(ConditionTable(config))
  local seq = nn.Sequential()
  seq:add(CGRU(config.kWordDim, config.kHiddenDim, config.kHiddenDim))
  --[[
  local w = LookupLinearNoBias(word_vocab:size(), config.kHiddenDim)
  w:share(lookup, 'weight', 'gradWeight')
  --]]
  local w = nn.LinearNoBias(config.kHiddenDim, word_vocab:size())
  seq:add(w)
  seq:add(nn.LogSoftMax())
  model:add(nn.Sequencer(seq))
  return model
end
