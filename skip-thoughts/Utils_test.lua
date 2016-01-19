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

mytester:add(UtilsTest)

mytester:run()


