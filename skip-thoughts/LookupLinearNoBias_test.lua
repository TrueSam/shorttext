package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("os")
require("rnn")
require("Config")
require("DataSet")
require("LookupLinearNoBias")
require("VocabularyBuilder")
require("VocabularyUtils")

mytester = torch.Tester()

local LookupLinearNoBiasTest = {}

function LookupLinearNoBiasTest.BatchTest()
  local config = Config()

  local batch = torch.Tensor{{1, 2, 3}, {2, 3, 4}}
  local table = LookupLinearNoBias(2, 3)
  table.weight = torch.Tensor{{1, 2, 3}, {4, 5, 6}}
  local output = table:forward(batch)
  assert(output ~= nil)
  assert(output:dim() == 2)
  assert(output:size(1) == 2)
  assert(output:size(2) == 2)
  assert(output[1][1] == 14)
  assert(output[1][2] == 32)

  assert(output[2][1] == 20)
  assert(output[2][2] == 47)
end

mytester:add(LookupLinearNoBiasTest)

mytester:run()



