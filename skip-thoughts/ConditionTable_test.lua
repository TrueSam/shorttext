package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("os")
require("ConditionTable")
require("Config")

mytester = torch.Tester()

local ConditionTableTest = {}

function ConditionTableTest.Test()
  local config = Config()
  config.useGPU = false
  local table = ConditionTable(config)
  local seq = {torch.randn(3), torch.randn(3), torch.randn(3)}
  local cond = torch.randn(3)
  local input = {seq, cond}
  local output = table:forward(input)
  assert(#output, 3);
  for i = 1, 3 do
    assert(#output[i], 2)
    assert(torch.eq(output[i][1], seq[i]))
    assert(torch.eq(output[i][2], cond))
  end
  local gradOutput = {{torch.randn(3), torch.randn(3)}, {torch.randn(3), torch.randn(3)}, {torch.randn(3), torch.randn(3)}}
  local gradInput = table:backward(input, gradOutput)
  assert(#gradInput, 2);
  assert(#gradInput[1], 3);
  for i = 1, 3 do
    assert(torch.eq(gradInput[1][i], gradOutput[i][1]))
  end
  assert(torch.eq(gradInput[2], torch.add(gradOutput[1][2], gradOutput[2][2], gradOutput[1][2])))
end

mytester:add(ConditionTableTest)

mytester:run()


