require("torch")

local VectorDataSet = torch.class("VectorDataSet")

VectorDataSet.COSINE = 0
VectorDataSet.EUCLIDEAN = 1

function VectorDataSet:__init(size, dim)
  assert(size > 0)
  assert(dim > 0)
  self.vectors_ = torch.Tensor(size, dim)
  self.labels_ = torch.IntTensor(size)
  self.nearest_ = torch.IntTensor(size)
  self.texts_ = {}
  self.original_texts_ = {}
end

function VectorDataSet:precision(distance_type)
  local size = self.labels_:size(1)
  assert(#self.texts_ == #self.original_texts_)
  assert(#self.texts_ == size)
  for i = 1, size do
    assert(self.labels_[i] == 1 or self.labels_[i] == -1)
    assert(self.texts_[i] ~= nil)
  end
  local correct = 0.0
  local positive = {}
  local negative = {}
  for i = 1, size do
    local m = self:argmin(i, distance_type)
    assert(m ~= nil)
    self.nearest_[i] = m
    if self.labels_[i] == self.labels_[m] then
      correct = correct + 1.0
      table.insert(positive, i)
    else
      table.insert(negative, i)
    end
  end
  print('Correct:')
  self:ShowExamples(positive)
  print('Wrong:')
  self:ShowExamples(negative)
  return correct / size
end

function VectorDataSet:ShowExamples(examples)
  local index = torch.randperm(#examples)
  n = math.min(3, #examples)
  for i = 1, n do
    local k = examples[index[i]]
    local m = self.nearest_[k]
    assert(k ~= nil and k >= 1 and k <= size)
    assert(m ~= nil and m >= 1 and m <= size)
    print('/:', self.labels_[k], self.texts_[k])
    print('|:', self.original_texts_[k])
    print('\\:', self.labels_[m], self.texts_[m])
    print('|:', self.original_texts_[m])
  end
end

function VectorDataSet:cosine(i, j)
  local n = self.vectors_[i]:norm()
  local s = self.vectors_[i]:dot(self.vectors_[j])
  local v = self.vectors_[j]:norm()
  local t = s / n / v
  return 1 - t
end

function VectorDataSet:euclidean(i, j)
  local n = self.vectors_[i] - self.vectors_[j]
  local t = n:norm()
  return t
end

function VectorDataSet:argmin(index, distance_type)
  assert(distance_type == VectorDataSet.COSINE or distance_type == VectorDataSet.EUCLIDEAN)

  local size = self.labels_:size(1)
  assert(size > 1)
  assert(index >= 1 and index <= size)
  local m = -1
  local d = 0
  for i = 1, size do
    if i ~= index then
      local t
      if distance_type == VectorDataSet.COSINE then
        t = self:cosine(index, i)
      elseif distance_type == VectorDataSet.EUCLIDEAN then
        t = self:euclidean(index, i)
      end
      if m >= 1 then
        if t < d then
          m = i
          d = t
        end
      else
        m = i
        d = t
      end
    end
  end
  assert(m >= 1 and m <= size and m ~= index)
  return m
end
