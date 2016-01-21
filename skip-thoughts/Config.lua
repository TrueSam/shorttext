require("torch")
require("nn")

local Config = torch.class("Config")


function Config:__init()
  -- The mini frequency for the text to keep in vocabulary
  self.kMinWordFreq = 100

  -- The size for the minibatch of sentences.
  -- Set batch size to bigger number if not using GPU.
  self.kBatchSize = 5
  self.kSentenceSize = -1
  self.kMaxSentenceSize = 100
  self.kNumBucket = 20

  -- The sample size for the dataset, -1 means using all the data.
  self.kSampleSize = 100000

  -- for validation
  self.kRTSampleSize = 500

  self.kWordDim = 50
  self.kHiddenDim = 50

  self.kPretrainedFile = "vectors-1billion-50.t7"
  -- self.kPretrainedFile = "glove.6B.50d.t7"

  -- Learning rate
  self.kLearningRate = 0.005
  self.kMinLearningRate = 0.001
  self.kLearningRateDecay = 0.0001

  self.kDataPath = "data"

  self.useGPU = false
end
