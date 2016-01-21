require('torch')
require("paths")
require("Config")
require("Utils")
require("ValidationDataSet")
require("Vocabulary")

local ManualDataSet, parent = torch.class('ManualDataSet', 'ValidationDataSet')

function ManualDataSet:__init(config, word_vocab)
  parent.__init(self)

  local filename = paths.concat(config.kDataPath, "manual.wrd")
  local lines = Utils.ReadLines(filename)
  assert(lines ~= nil)
  local num_tokens = Utils.CountTokens(lines)
  local num_lines = #lines
  assert(num_lines > 0 and num_lines % 2 == 0)
  assert(num_tokens > 0)
  self.tokens_ = torch.IntTensor(num_tokens)
  self.starts_ = torch.IntTensor(num_lines)
  self.lengths_ = torch.IntTensor(num_lines)
  self.labels_ = torch.IntTensor(num_lines)
  self.texts_ = {}
  self.sentences_ = {}

  local token = 1
  local sentence = 1

  local labels = {1, -1}
  for k = 1, #lines do
    local line = lines[k]
    if sentence > num_lines then
      break
    end
    assert(token <= num_tokens)
    local parts = Utils.split(line)
    assert(#parts > 0)
    self.lengths_[sentence] = #parts
    self.starts_[sentence] = token

    local s = {}
    for i = 1, #parts do
      local id = word_vocab:word_id(parts[i])
      local text = word_vocab:word_text(id)
      self.tokens_[token] = id
      table.insert(s, text)
      token = token + 1
    end

    self.texts_[sentence] = table.concat(s, " ")
    self.sentences_[sentence] = line

    sentence = sentence + 1
  end

  assert(num_lines + 1 == sentence)
  assert(num_tokens + 1 == token)
  assert(num_lines == #self.sentences_)
  for i = 1, num_lines, 2 do
    self.labels_[i] = i + 1
    self.labels_[i + 1] = i
  end
end

function ManualDataSet:Examples(nearest)
  local e = {}
  local p = 0
  local n = 0
  local size = self.lengths_:size(1)
  for i = 1, size do
    if self.labels_[i] == nearest[i] then
      if p < 3 then
        table.insert(e, {1, i, nearest[i]})
      end
      p = p + 1
    else
      if n < 3 then
        table.insert(e, {-1, i, nearest[i]})
      end
      n = n + 1
    end
  end
  return e
end

function ManualDataSet:Precision(nearest)
  assert(nearest:size(1) == self.lengths_:size(1))
  local correct = 0
  local size = nearest:size(1)
  for i = 1, size do
    if self.labels_[i] == nearest[i] then
      correct = correct + 1
    end
  end
  return correct / size
end

function ManualDataSet:Show(examples)
  ValidationDataSet.Show(examples, self.lengths_:size(1), self.texts_, self.sentences_)
end
