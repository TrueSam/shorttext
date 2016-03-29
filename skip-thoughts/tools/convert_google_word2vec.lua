require('torch')
require('pl')

assert(#arg == 2)

local max_w = 50

local infile = io.open(arg[1], 'r')

local words = tonumber(infile:read("*l"))
local size = tonumber(infile:read("*l"))

print('There are ' .. words .. ' tokens, ' .. size .. ' dimension')

assert(words > 0)
assert(size > 0)

local filebase, shards = utils.splitv(arg[2], '@')
assert(filebase ~= nil)
assert(shards ~= nil)
shards = shards or '1'
shards = tonumber(shards)
assert(shards > 0)

local length
if words % shards == 0 then
  length = words / shards
else
  length = words / shards + 1
end

for j = 1, shards do
  local word2vec = {}

  word2vec['embeddings'] = torch.FloatTensor(length, size)
  word2vec['idMap'] = {}

  for i = 1, length do
    local s = (j - 1) * length + i

    if i % 1000 == 0 then
      -- print('Reading ' .. i .. ' lines.')
    end
    if s > words then
      break
    end
    local line = infile:read("*l")
    assert(line:len() > 0)
    local parts = {}
    for w in string.gmatch(line, "%S+") do
      table.insert(parts, w)
    end
    assert(#parts == size + 1, line)
    local w = parts[1]:gsub("_", " ")

    word2vec.idMap[w] = i
    for k = 1, size do
      word2vec.embeddings[i][k] = tonumber(parts[k + 1])
    end
  end
  local filename = string.format('%s.t7-%05d-of-%05d', filebase, j - 1, shards)
  print('saving ' .. filename)
  torch.save(filename, word2vec)
  collectgarbage()
end
