require("torch")
require("Config")
local utf8 = require("lua-utf8")

local Utils = torch.class("Utils")

-- Utility functions
Utils.split = function(text)
  local parts = {}
  for w in utf8.gmatch(text, "%S+") do
    table.insert(parts, w)
  end
  return parts
end

Utils.explode = function (text)
  local chars = {}
  for i = 1, utf8.len(text) do
    table.insert(chars, utf8.sub(text, i, i))
  end
  return chars
end

Utils.mean = function(a)
  assert(a ~= nil)
  local mean = 0.0
  for i = 1, #a do
    mean = mean + a[i]
  end
  if #a > 0 then
    mean = mean / #a
  end
  return mean
end

Utils.precision = function(prediction, truth)
  assert(prediction ~= nil)
  assert(truth ~= nil)
  assert(#prediction == #truth)
  local c = 0.0
  local l = 0
  for i = 1, #prediction do
    assert(prediction[i]:dim() == 2)
    assert(truth[i]:dim() == 1)
    assert(prediction[i]:size(1) == truth[i]:size(1), prediction[i]:size(1) .. " vs. " .. truth[i]:size(1))
    assert(prediction[i]:size(2) == 1)
    assert(prediction[i]:nElement() == truth[i]:nElement())
    local p = prediction[i]
    local t = truth[i]
    local len = p:size(1)
    l = l + len
    for k = 1, len do
      assert(p[k][1] ~= nil)
      assert(t[k] ~= nil)
      if p[k][1] == t[k] then
        c = c + 1.0
      end
    end
  end
  return c / l
end

function Utils.ReadLines(in_file)
  assert(type(in_file) == "string")
  local lines = {}
  for line in io.lines(in_file) do
    table.insert(lines, line)
  end
  return lines
end

function Utils.CountTokens(lines)
  local num_tokens = 0
  for _, line in ipairs(lines) do
    local parts = Utils.split(line)
    assert(#parts > 0)
    num_tokens = num_tokens + #parts
  end
  return num_tokens
end

function Utils.ChunkSentence(parts, chunk)
  local size = #parts
  if chunk > 0 then
    if size > chunk then
      size = chunk
    end
  end
  return size
end

function Utils.CountTokensAndLines(config, in_file)
  local size = config.kSampleSize
  local chunk = config.kSentenceSize
  assert(type(in_file) == "string")
  assert(type(size) == 'number')
  assert(type(chunk) == 'number')
  local num_tokens = 0
  local num_lines = 0
  local i = 1
  for line in io.lines(in_file) do
    if size >= 0 and i > size then
      break
    end
    local parts = Utils.split(line)
    assert(#parts > 0)
    if #parts <= config.kMaxSentenceSize then
      local tokens = Utils.ChunkSentence(parts, chunk)
      assert(tokens > 0)
      num_tokens = num_tokens + tokens
      num_lines = num_lines + 1
      i = i + 1
    end
  end
  return num_tokens, num_lines
end

function Utils.GetBucketIndex(limits, n)
  for k = 1, #limits do
    if n <= limits[k] then
      return k
    end
  end
end

function Utils.BuildBuckets(lengths, num_buckets)
  assert(num_buckets > 0)
  local sorted = torch.sort(lengths, 1)
  local u = 0
  local unique = {}
  local unique_count = {}
  for i = 1, sorted:size(1) do
    if i - 1 >= 1 then
      if sorted[i] ~= sorted[i - 1] then
        table.insert(unique, sorted[i])
        table.insert(unique_count, 1)
      else
        unique_count[#unique_count] = unique_count[#unique_count] + 1
      end
    else
      table.insert(unique, sorted[i])
      table.insert(unique_count, 1)
    end
  end
  local len = math.floor(lengths:size(1) / num_buckets)
  local buckets = {}
  local u = 0
  for i = 1, #unique_count do
    if #buckets >= num_buckets - 1 then
      break
    end
    if u + unique_count[i] >= len then
      table.insert(buckets, unique[i])
      u = 0
    else
      u = u + unique_count[i]
    end
  end
  table.insert(buckets, unique[#unique])
  assert(#buckets <= num_buckets)
  for i = 1, #buckets - 1 do
    assert(buckets[i] < buckets[i + 1])
  end
  return buckets
end

function Utils.AssembleIndex(buckets, index)
  assert(#buckets == #index)
  local n = 0
  for i = 1, #index do
    assert(index[i] > 0 and index[i] <= buckets[i])
    n = n * buckets[i] + index[i] - 1
  end
  return n + 1
end

function Utils.DisassembleIndex(buckets, index)
  local retv = {}
  index = index - 1
  for i = #buckets, 1, -1 do
    local n = index % buckets[i]
    retv[i] = n
    index = math.floor(index / buckets[i])
  end
  assert(index == 0)
  for i = 1, #retv do
    assert(retv[i] >= 0 and retv[i] < buckets[i])
    retv[i] = retv[i] + 1
  end
  return retv
end

function Utils.RandomPermute(array)
  local index = torch.randperm(#array)
  local p = {}
  for i = 1, index:size(1) do
    local v = array[index[i]]
    assert(v ~= nil)
    table.insert(p, v)
  end
  return p
end

function Utils.MergeBuckets(buckets, min_size)
  local small_buckets = {}
  local large_buckets = {}
  for i = 1, #buckets do
    if #buckets[i] > 0 then
      if #buckets[i] < min_size then
        table.insert(small_buckets, i)
      else
        table.insert(large_buckets, i)
      end
    end
  end
  local function merge(from, to)
    assert(#from > 0)
    assert(#to > 0)
    for i = 1, #from do
      table.insert(to, from[i])
    end
    from = {}
  end
  for i = 2, #small_buckets do
    merge(buckets[small_buckets[i]], buckets[small_buckets[1]])
  end
  table.insert(large_buckets, small_buckets[1])

  local p = {}
  for i = 1, #large_buckets do
    assert(#buckets[large_buckets[i]] > 0)
    table.insert(p, torch.IntTensor(buckets[large_buckets[i]]))
  end
  return p
end

function Utils.DistributeBuckets(lengths, limits)
  local index = {}
  for i = 1, #limits do
    for j = 1, #limits do
      for k = 1, #limits do
        table.insert(index, {})
      end
    end
  end
  local buckets = {#limits, #limits, #limits}
  for i = 1, lengths:size(1) - 2 do
    local p = Utils.GetBucketIndex(limits, lengths[i])
    local c = Utils.GetBucketIndex(limits, lengths[i + 1])
    local n = Utils.GetBucketIndex(limits, lengths[i + 2])
    assert(p >= 1 and p <= #limits)
    assert(c >= 1 and c <= #limits)
    assert(n >= 1 and n <= #limits)
    local idx = {p, c, n}
    local b = Utils.AssembleIndex(buckets, idx)
    local t = index[b]
    assert(t ~= nil)
    table.insert(t, i)
  end
  for i = 1, #index do
    if #index[i] > 0 then
      index[i] = Utils.RandomPermute(index[i])
    end
  end
  return index
end


-- End Utility fucntions

