require('torch')

assert(#arg == 2)

local max_w = 50

local infile = io.open(arg[1], 'r')

local words = tonumber(infile:read("*l"))
local size = tonumber(infile:read("*l"))

print('There are ' .. words .. ' tokens, ' .. size .. ' dimension')

assert(words > 0)
assert(size > 0)

local word2vec = {}

word2vec['embeddings'] = torch.FloatTensor(words, size)
word2vec['idMap'] = {}

for i = 1, words do
  if i % 1000 == 0 then
    print('Reading ' .. i .. ' lines.')
  end
  local line = infile:read("*l")
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

torch.save(arg[2], word2vec)
