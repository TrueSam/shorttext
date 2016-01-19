require("torch")
require("Model")
require("VectorDataSet")
require("PolarityDataSet")

local TrainUtils = torch.class("TrainUtils")

function TrainUtils.PolarityValidation(model, rt_dataset, distance_type)
  local dev_loss = {}
  local dev_pred = {}
  local dev_tag = {}
  local P = rt_dataset:size()
  local dev_dataset = VectorDataSet(P, model.config_.kHiddenDim)
  model.model_:evaluate()
  for j = 1, P do
    local sentence, label, text, original = rt_dataset:GetSentence(j)
    dev_dataset.vectors_[j]:copy(model:vector(sentence):float())
    dev_dataset.labels_[j] = label
    dev_dataset.texts_[j] = text
    dev_dataset.original_texts_[j] = original
  end
  model.model_:training()
  return 1 - dev_dataset:precision(distance_type)
end
