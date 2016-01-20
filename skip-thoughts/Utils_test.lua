package.path = package.path .. ";./?.lua"

require("torch")
require("nn")
require("rnn")
require("os")
require("Utils")
require("Config")

mytester = torch.Tester()

local UtilsTest = {}

function UtilsTest.Test()
  local line = 'ab c d e f g h i j k'
  local parts = Utils.split(line)
  assert(#parts == 10)
  local tokens = Utils.ChunkSentence(parts, 3)
  assert(tokens == 3)
  tokens = Utils.ChunkSentence(parts, 30)
  assert(tokens == 10)
end

function UtilsTest.BucketsTest()
  local lengths = torch.IntTensor{1, 1, 3, 4, 2, 3, 5, 4, 5, 2}
  local limits = Utils.BuildBuckets(lengths, 3)
  assert(#limits == 3)
  assert(limits[1] == 2)
  assert(limits[2] == 4)
  assert(limits[3] == 5)
  local buckets = Utils.DistributeBuckets(lengths, limits)
  assert(#buckets == 27)
  assert(#buckets[2] > 0)
  assert(buckets[2][1] == 1)

  local index = Utils.AssembleIndex(limits, {1, 1, 3})
  local indice = Utils.DisassembleIndex(limits, index)
  assert(#indice == 3)
  assert(indice[1] == 1)
  assert(indice[2] == 1)
  assert(indice[3] == 3)

  limits = Utils.BuildBuckets(lengths, 1)
  buckets = Utils.DistributeBuckets(lengths, limits)
  assert(#buckets == 1)


  local lengths = torch.IntTensor{13, 11, 12, 35, 25, 7, 2}
  local limits = Utils.BuildBuckets(lengths, 3)
  assert(#limits == 3)
  assert(limits[1] == 7)
  assert(limits[2] == 12)
  assert(limits[3] == 35)
end

function UtilsTest.PermTest()
  local arr = {}
  for i = 1, 1000 do
    table.insert(arr, i)
  end
  local result = Utils.RandomPermute(arr)
  local identical = true
  for i = 1, 1000 do
    if arr[i] ~= result[i] then
      identical = false
    end
  end
  assert(identical == false)
end

mytester:add(UtilsTest)

mytester:run()


