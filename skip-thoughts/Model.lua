require("torch")
require("nn")
require("pl")
require("Config")
require("Encoder")
require("Decoder")
require("CriterionTable")

local Model = torch.class("Model")

function Model:__init(config, word_vocab)
  self.config_ = config
  if config.useGPU == true then
    require("cutorch")
    require("cunn")
    require("cudnn")
    require("cunnx")
  end

  self.model_ = nn.Sequential()

  -- input is table of size 3 {prev_sentence, current_sentence, next_sentence}
  local seq = nn.ParallelTable()

  local e0 = nn.LookupTable(word_vocab:size(), config.kWordDim)
  local e1 = nn.LookupTable(word_vocab:size(), config.kWordDim)
  local e2 = nn.LookupTable(word_vocab:size(), config.kWordDim)

  local filename = paths.concat(config.kDataPath, config.kPretrainedFile)
  self:_LoadPretrainedEmbeddings(filename, word_vocab, e0)

  -- share weights
  e1:share(e0, "weight", "gradWeight")
  e2:share(e0, "weight", "gradWeight")

  self.encoder_ = Encoder.create(config, e1)
  seq:add(Encoder.createSequence(config, e0)):add(self.encoder_):add(Encoder.createSequence(config, e2))
  self.model_:add(seq)  -- prev, h(curr), next

  -- Rearrange to prev, h(curr), next, h(curr)
  local concat = nn.ConcatTable()
  concat:add(nn.SelectTable(1)):add(nn.SelectTable(2)):add(nn.SelectTable(3)):add(nn.SelectTable(2))
  self.model_:add(concat)

  local concat = nn.ConcatTable()
  local t1 = nn.Sequential():add(nn.NarrowTable(1, 2)):add(Decoder.create(config, word_vocab, e0))
  local t2 = nn.Sequential():add(nn.NarrowTable(3, 2)):add(Decoder.create(config, word_vocab, e0))
  concat:add(t1):add(t2)
  self.model_:add(concat)

  self.criterion_ = CriterionTable(nn.SequencerCriterion(nn.ClassNLLCriterion()))

  if config.useGPU == true then
    self.model_ = self.model_:cuda()
    self.criterion_ = self.criterion_:cuda()
  end

  self.eventCount = 0
end

function Model:WriteToFile(filename)
  if self.config_.useGPU == true then
    self.model_ = self.model_:float()
    self.criterion_ = self.criterion_:float()
  end

  torch.save(filename, self)
end

function Model:ReadFromFile(filename)
  self = torch.load(filename)
  self.eventCount = 0
  if self.config_.useGPU == true then
    self.model_ = self.model_:cuda()
    self.criterion_ = self.criterion_:cuda()
  end
end

function Model:train(sentence_triple, label_pair)
  assert(#sentence_triple == 3)
  assert(#label_pair == 2)
  assert(#sentence_triple[1] == #label_pair[1])
  assert(#sentence_triple[3] == #label_pair[2])
  assert(#sentence_triple[1] > 0)
  assert(#sentence_triple[2] > 0)
  assert(#sentence_triple[3] > 0)
  for k = 1, #sentence_triple do
    assert(#sentence_triple[k] > 0)
    for i = 1, #sentence_triple[k] do
      assert(sentence_triple[k][1]:size(1) == sentence_triple[k][i]:size(1))
    end
  end
  for k = 1, #label_pair do
    assert(#label_pair[k] > 0)
    for i = 1, #label_pair[k] do
      assert(label_pair[k][1]:size(1) == label_pair[k][i]:size(1))
    end
  end

  collectgarbage()

  local output = self.model_:forward(sentence_triple)
  local loss = self.criterion_:forward(output, label_pair)
  local grad = self.criterion_:backward(output, label_pair)
  assert(grad ~= nil)
  assert(loss ~= nil)
  assert(output ~= nil)

  local nevals = self.eventCount
  local lr = self.config_.kLearningRate
  local lrd = self.config_.kLearningRateDecay
  local clr = lr / (1 + nevals * lrd)
  assert(clr >= 0.0)

  self.model_:zeroGradParameters()
  self.model_:backward(sentence_triple, grad)
  self.model_:updateParameters(clr)
  self.eventCount = self.eventCount + 1
  return loss
end

function Model:vector(sentence)
  assert(sentence:dim() == 1)
  assert(sentence:size(1) > 0)
  local batch = {}
  for i = 1, sentence:size(1) do
    table.insert(batch, sentence:narrow(1, i, 1):clone())
  end
  if self.config_.useGPU == true then
    for i = 1, #batch do
      batch[i] = batch[i]:cuda()
    end
  end
  local v = self.encoder_:forward(batch)
  assert(v:dim() == 2)
  assert(v:size(1) == 1)
  -- The input is batch input, so we select the vector.
  return v:select(1, 1)
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
