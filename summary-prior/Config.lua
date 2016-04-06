require("torch")
require("nn")

local Config = torch.class("Config")


function Config:__init()
  -- The mini frequency for the text to keep in vocabulary
  self.kMinWordFreq = 0

  -- The size for the minibatch of sentences.
  -- Set batch size to bigger number if not using GPU.
  self.kBatchSize = 1
  self.kSentenceSize = -1
  self.kMaxSentenceSize = 100
  self.kMinSentenceSize = 5
  self.kNumBucket = 20

  -- The sample size for the dataset, -1 means using all the data.
  self.kSampleSize = -1

  -- for validation
  self.kRTSampleSize = 500

  self.kWordDim = 50
  self.kSentenceFeatureDim = 75
  self.kDocumentFeatureDim = 3
  self.kWindowSize = 3

  self.kPretrainedFile = "vectors-1billion-50.t7"

  -- Learning rate
  self.kLearningRate = 0.05
  self.kMinLearningRate = 0.001
  self.kLearningRateDecay = 0.0001

  self.kDataPath = "data"

  self.useGPU = false
end
