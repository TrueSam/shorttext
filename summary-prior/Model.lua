require('torch')
require('nn')

local Model = torch.class('Model')

function Model:__init(config, word_vocab)
  self.model_ = nn.Sequential()
  self.model_:add(nn.ParallelTable():add(self:createSentenceModel(config, word_vocab)):add(self:createDocumentModel()))
  self.model_:add(nn.JoinTable(2))
  self.model_:add(nn.Linear(config.kSentenceFeatureDim + config.kDocumentFeatureDim, 1))

  self.criterion_ = nn.MSECriterion()

  self.config_ = config
end

function Model:createSentenceModel(config, word_vocab):
  -- We always assume the input is mini-batched.
  local model = nn.Squential()
  model:add(nn.LookupTable(word_vocab:size(), config.kWordDim))
  local c = nn.ConcatTable()
  for kW = 1, config.kWindowSize do
    local cnn = nn.Squential()
    cnn:add(nn.TemporalConvolution(config.kWordDim, config.kSentenceFeatureDim, kW))
    cnn:add(nn.Max(2))
    c:add(cnn)
  end
  model:add(c)
  model:add(nn.JoinTable(2)):add(nn.Max(2))
  return model
end

function Model:createDocumentModel():
  local model = nn.Sequential()
  model:add(nn.Identity())
  return model
end

function Model:train(input)
  assert(type(input) == 'table')
  assert(#input == 3)
  -- mini batch input.
  assert(input[1]:dim() == 2)
  assert(input[2]:dim() == 2)
  assert(input[1]:size(1) == input[2]:size(1))
  assert(input[2]:size(2) == self.config_.kDocumentFeatureDim)
  assert(input[3]:dim() == 1)
  assert(input[3]:size(1) == input[1]:size(1))

  local data = {input[1], input[2]}
  local target = input[3]

  local output = self.model_:forward(data)
  self.criterion_:forward(output, target)
  local dp = self.criterion_:backward(output, target)
  self.model_:backward(data, dp)

end

function Model:evaluate()
end
