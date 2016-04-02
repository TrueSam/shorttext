require("torch")
require("Config")
require("VocabularyUtils")

local Vocabulary = torch.class("Vocabulary")

function Vocabulary:__init(unknown_id, count, freq)
  assert(type(unknown_id) == "number")
  assert(count ~= nil)
  assert(freq ~= nil)
  self.unknown_ = unknown_id
  self.word2id_ = count
  self.id2word_ = {}
  self.id2freq_ = freq
  for k, v in pairs(count) do
    self.id2word_[v] = k
  end
  local tmp = {}
  for k, v in pairs(self.id2word_) do
    table.insert(tmp, k)
  end
  table.sort(tmp)
  for i = 1, #tmp - 1 do
    assert(tmp[i] == tmp[i + 1] - 1, tmp[i] .. " " .. tmp[i + 1])
  end
  self.num_word_ = #tmp
end

function Vocabulary:size()
  return self.num_word_
end

function Vocabulary:word_text(id)
  assert(id ~= nil)
  local text = self.id2word_[id]
  assert(text ~= nil)
  return text
end

function Vocabulary:word_id(text)
  assert(text ~= nil)
  local id = self.word2id_[text]
  if id == nil then
    id = self.unknown_
  end
  return id
end

function Vocabulary:text_count(text)
  return self:id_count(self:word_id(text))
end

function Vocabulary:id_count(id)
  local freq = self.id2freq_[id]
  assert(freq ~= nil)
  return freq.count
end

function Vocabulary:text_category_count(text)
  return self:id_category_count(self:word_id(text))
end

function Vocabulary:id_category_count(id)
  local freq = self.id2freq_[id]
  assert(freq ~= nil)
  return freq.category_count
end
