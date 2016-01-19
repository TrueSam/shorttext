require("torch")
require("os")

require("Config")
require("Utils")
require("Vocabulary")
require("VocabularyUtils")

local utf8 = require("lua-utf8")

local VocabularyBuilder = torch.class("VocabularyBuilder")

function VocabularyBuilder:__init(config)
  assert(type(config.kMinWordFreq) == "number")
  self.config_ = config
end

function VocabularyBuilder:BuildWordVocabulary(in_file)
  local scored_count, count_freq = VocabularyUtils.ReadWordVocabulary(in_file, self.config_.kMinWordFreq, false)
  return Vocabulary(VocabularyUtils.kUnknownWordId, scored_count, count_freq)
end
