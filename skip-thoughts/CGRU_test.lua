package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("os")
require("CGRU")
require("Config")

mytester = torch.Tester()

local CGRUTest = {}

function CGRUTest.Test()
  local config = Config()
  local table = nn.Sequencer(CGRU(config.kWordDim, config.kHiddenDim))
  local cond = torch.randn(2, config.kHiddenDim)
  local input = {{torch.randn(2, config.kWordDim), cond:clone()},
  {torch.randn(2, config.kWordDim), cond:clone()},
  {torch.randn(2, config.kWordDim), cond:clone()}}
  local output = table:forward(input)
  assert(#output == #input)
  for i = 1, #output do
    assert(output[i]:size(1) == 2)
    assert(output[i]:size(2) == config.kHiddenDim)
  end
end

mytester:add(CGRUTest)

mytester:run()


