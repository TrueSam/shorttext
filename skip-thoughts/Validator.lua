require('torch')

local Validator = torch.class('Validator')

Validator.COSINE = 0
Validator.EUCLIDEAN = 1

function Validator:__init(config, tokens, starts, lengths)
  assert(config ~= nil)
  assert(lengths ~= nil)
  assert(starts ~= nil)
  assert(tokens ~= nil)
  assert(lengths:size(1) == starts:size(1))
  self.lengths_ = lengths
  self.tokens_ = tokens
  self.starts_ = starts
  self.vectors_ = torch.Tensor(lengths:size(1), config.kHiddenDim)
end

function Validator:Cosine(i, j)
  local n = self.vectors_[i]:norm()
  local s = self.vectors_[i]:dot(self.vectors_[j])
  local v = self.vectors_[j]:norm()
  local t = s / n / v
  return 1 - t
end

function Validator:Euclidean(i, j)
  local n = self.vectors_[i] - self.vectors_[j]
  local t = n:norm()
  return t
end

function Validator:Argmin(index, distance_type)
  assert(distance_type == Validator.COSINE or distance_type == Validator.EUCLIDEAN)

  local size = self.lengths_:size(1)
  assert(size > 1)
  assert(index >= 1 and index <= size)
  local m = -1
  local d = 0
  for i = 1, size do
    if i ~= index then
      local t
      if distance_type == Validator.COSINE then
        t = self:Cosine(index, i)
      elseif distance_type == Validator.EUCLIDEAN then
        t = self:Euclidean(index, i)
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

function Validator:FindNearest(model, distance_type)
  local nearest = torch.IntTensor(self.lengths_:size(1))
  local size = self.lengths_:size(1)
  model.model_:evaluate()
  for i = 1, size do
    local sentence = self.tokens_:narrow(1, self.starts_[i], self.lengths_[i])
    self.vectors_[i]:copy(model:vector(sentence):float())
  end
  for i = 1, size do
    local m = self:Argmin(i, distance_type)
    assert(m ~= nil)
    nearest[i] = m
  end
  model.model_:training()
  return nearest
end
