package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("rnn")
require("os")
require("CriterionTable")
require("Config")

mytester = torch.Tester()

local CriterionTableTest = {}

function CriterionTableTest.Test()
  local input0 = {
    torch.randn(2, 5),
    torch.randn(2, 5)
  }
  local target0 = {
    torch.IntTensor{1, 2},
    torch.IntTensor{2, 3},
  }
  local input1 = {
    torch.randn(2, 5),
    torch.randn(2, 5),
    torch.randn(2, 5)
  }
  local target1 = {
    torch.IntTensor{1, 2},
    torch.IntTensor{2, 3},
    torch.IntTensor{3, 4}
  }
  local input = {input0, input1}
  local target = {target0, target1}
  local c = nn.ClassNLLCriterion()
  local loss
  local s = nn.SequencerCriterion(c)
  local m = CriterionTable(s)
  loss = m:forward(input, target)
  assert(loss ~= nil)
  local dp = m:backward(input, target)
  assert(#dp == #input)
  for i = 1, #dp do
    assert(#dp[i] == #input[i])
    for k = 1, #dp[i] do
      assert(dp[i][k]:dim() == input[i][k]:dim())
      assert(dp[i][k]:dim() == 2)
      assert(dp[i][k]:size(1) == input[i][k]:size(1))
      assert(dp[i][k]:size(2) == input[i][k]:size(2))
    end
  end
end

mytester:add(CriterionTableTest)

mytester:run()


