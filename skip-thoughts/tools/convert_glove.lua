require('torch')
require('pl.utils')

assert(#arg == 2)

local function split(line)
  local parts = {}
  for w in string.gmatch(line, "%S+") do
    table.insert(parts, w)
  end
  return parts
end

local max_w = 50

-- local infile = io.open(arg[1], 'r')

local words = 0
local size = 0

for line in io.lines(arg[1]) do
  words = words + 1
  if size == 0 then
    size = #split(line) - 1
  end
end

print('There are ' .. words .. ' tokens, ' .. size .. ' dimension')

assert(words > 0)
assert(size > 0)

local word2vec = {}

word2vec['embeddings'] = torch.FloatTensor(words, size)
word2vec['idMap'] = {}

local i = 1
for line in io.lines(arg[1]) do
  if i % 1000 == 0 then
    print('Reading ' .. i .. ' lines.')
  end
  local parts = split(line)
  assert(#parts == size + 1, line)
  local w = parts[1]:gsub("_", " ")

  word2vec.idMap[w] = i
  for k = 1, size do
    word2vec.embeddings[i][k] = tonumber(parts[k + 1])
  end
  i = i + 1
end

torch.save(arg[2], word2vec)
