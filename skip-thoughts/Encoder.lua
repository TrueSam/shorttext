require("rnn")
require("nn")
require("torch")

local Encoder = torch.class("Encoder")

function Encoder.createSequence(config, lookup)
  local model = nn.Sequential()
  model:add(nn.Sequencer(lookup))
  return model
end

function Encoder.create(config, lookup)
  local model = nn.Sequential()
  model:add(Encoder.createSequence(config, lookup))
  model:add(nn.Sequencer(nn.GRU(config.kWordDim, config.kHiddenDim)))
  model:add(nn.SelectTable(-1))
  return model
end


