require('torch')
require('nn')
require('optim')
require('Validator')

local Model = torch.class('Model')

function Model:__init(config, word_vocab)
  assert(config ~= nil)
  assert(word_vocab ~= nil)

  self.lookup_ = nn.LookupTable(word_vocab:size(), config.kWordDim)
  local filename = paths.concat(config.kDataPath, config.kPretrainedFile)
  self:_LoadPretrainedEmbeddings(filename, word_vocab, self.lookup_)

  self.model_ = nn.Sequential()
  self.model_:add(nn.ParallelTable():add(self:createSentenceModel(config, word_vocab)):add(self:createDocumentModel()))
  self.model_:add(nn.JoinTable(2))
  self.model_:add(nn.Linear(config.kSentenceFeatureDim + config.kDocumentFeatureDim, 1))

  self.criterion_ = nn.MSECriterion()

  self.word_vocab_ = word_vocab

  self.config_ = config
end

function Model:createSentenceModel(config, word_vocab)
  -- We always assume the input is mini-batched.
  local model = nn.Sequential()
  model:add(self.lookup_)
  local c = nn.ConcatTable()
  for kW = 1, config.kWindowSize do
    local cnn = nn.Sequential()
    cnn:add(nn.TemporalConvolution(config.kWordDim, config.kSentenceFeatureDim, kW))
    cnn:add(nn.Max(2)):add(nn.Replicate(1, 2))
    c:add(cnn)
  end
  model:add(c)
  model:add(nn.JoinTable(1, 2)):add(nn.Max(1, 2))
  return model
end

function Model:createDocumentModel()
  local model = nn.Sequential()
  model:add(nn.Identity())
  return model
end

function Model:train(sampler, epochs)
  self.model_:training()
  x, dl_dx = self.model_:getParameters()
  feval = function(x_new)
    -- set x to x_new, if differnt
    -- (in this simple example, x_new will typically always point to x,
    -- so the copy is really useless)
    if x ~= x_new then
      x:copy(x_new)
    end

    local batch = sampler:SampleBatch()
    assert(#batch == 3)
    local sentences, features, targets
    local sentences = batch[1]
    local features = batch[2]
    local targets = batch[3]
    assert(sentences ~= nil)
    assert(features ~= nil)
    assert(targets ~= nil)

    assert(sentences:dim() == 2)
    assert(features:dim() == 2)
    assert(targets:dim() == 2)

    -- reset gradients (gradients are always accumulated, to accommodate 
    -- batch methods)
    dl_dx:zero()

    local inputs = {sentences, features}
    local output = self.model_:forward(inputs)
    -- evaluate the loss function and its derivative wrt x, for that sample
    local loss_x = self.criterion_:forward(output, targets)
    self.model_:backward(inputs, self.criterion_:backward(output, targets))

    -- clip gradient element-wise
    dl_dx:clamp(-5, 5)
    -- return loss(x) and dloss/dx
    return loss_x, dl_dx
  end

  sgd_params = {
    learningRate = 1,
    learningRateDecay = 1e-1,
    weightDecay = 0.0,
    momentum = 0.0
  }

  for e = 1, epochs do
    collectgarbage()
    local size = sampler:size() / self.config_.kBatchSize
    assert(size > 0)
    local current_loss = 0
    for i = 1, size do
      _, fs = optim.sgd(feval, x, sgd_params)
      -- handle early stopping if things are going really bad
      if fs[1] ~= fs[1] then
        print('loss is NaN.  This usually indicates a bug.')
        break -- halt
      end

      current_loss = current_loss + fs[1]

      if i % 100 == 0 then
        collectgarbage()
      end
    end

    if e % 100 == 0 then
      -- Limit the summary to 10 words.
      Validator.Validate(self, 10, e)
    end

    current_loss = current_loss / size
    print('current loss = ' .. current_loss .. ' at epoch ' .. e)
  end
end

function Model:evaluate(inputs)
  return self.model_:forward(inputs)
end

function Model:_LoadPretrainedEmbeddings(pretrained_filename, word_vocab, word_embedding)
  -- Load pre-trained word embedding
  print("Loading pre-trained word embeddings...")
  local word2vec = torch.load(pretrained_filename)
  print("Normalizing pre-trained word embeddings...")
  -- Normalize word2vec
  local mean = word2vec.embeddings:mean(2):expand(word2vec.embeddings:size())
  local std = word2vec.embeddings:std(2):expand(word2vec.embeddings:size())
  word2vec.embeddings = word2vec.embeddings - mean
  word2vec.embeddings:cdiv(std)
  local updated = {}
  for word, id in pairs(word2vec.idMap) do
    local i = word_vocab:word_id(word)
    if i == VocabularyUtils.kUnknownWordId then
      i = word_vocab:word_id(word:lower())
    end
    if i ~= VocabularyUtils.kUnknownWordId then
      assert(word_embedding.weight[i]:nElement() == word2vec.embeddings[id]:nElement())
      table.insert(updated, i)
      word_embedding.weight[i]:copy(word2vec.embeddings[id])
    end
  end
  local loaded_embeddings = Set.len(Set(updated))
  assert(loaded_embeddings > 0)
  print(loaded_embeddings .. " embeddings loaded.")
  print("Embedding coverage: " .. (loaded_embeddings / word_vocab:size()))
end
