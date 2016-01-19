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


-- End Utility fucntions

