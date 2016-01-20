package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("rnn")
require("os")
require("VectorDataSet")
require("Config")

mytester = torch.Tester()

local VectorDataSetTest = {}

function VectorDataSetTest.Test()
  local table = VectorDataSet(3, 2)
  table.vectors_[1]:copy(torch.Tensor{0, 0});
  table.vectors_[2]:copy(torch.Tensor{3, 0});
  table.vectors_[3]:copy(torch.Tensor{3, 4});
  table.labels_[1] = 1
  table.labels_[2] = 1
  table.labels_[3] = -1

  local m = table:argmin(1, VectorDataSet.EUCLIDEAN)
  assert(m == 2)

  m = table:precision(VectorDataSet.EUCLIDEAN, false)
  assert(math.abs(m - 2.0 / 3) <= 1E4)

  table.vectors_[1]:copy(torch.Tensor{1, 1});
  table.vectors_[2]:copy(torch.Tensor{2, 1});
  table.vectors_[3]:copy(torch.Tensor{3, 1});

  m = table:argmin(1, VectorDataSet.COSINE)
  assert(m == 2)

  m = table:argmin(2, VectorDataSet.COSINE)
  assert(m == 3)

end

mytester:add(VectorDataSetTest)

mytester:run()


