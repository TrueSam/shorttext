require('torch')
require('nn')
require('DataSet')
require('Utils')
require('pl.utils')
require('pl.stringx')
require('pl.file')

local Summarizer = torch.class('Summarizer')

function Summarizer:__init(model, limit)
  assert(limit > 0)
  assert(model ~= nil)
  self.model_ = model
  self.limit_ = limit
end

function Summarizer:GenerateSummary(sentences)
  local tokenized = self:_GetTokenizedSentences(sentences)
  local features = self:_GetSentenceFeatures(tokenized)
  local scores = self:_GetSentenceScores(features)
  local _, rerank = torch.sort(scores, 1, true)
  local summary = {}
  for i = 1, rerank:size(1) do
    local sentence = tokenized[rerank[i]]
    if self:IsSkip(summary, sentence) == false then
      table.insert(summary, sentence)
    end
  end
  return summary
end

function Summarizer:GenerateSummaryFromFile(filename)
  local document = file.read(filename)
  return self:GenerateSummaryText(document)
end

function Summarizer:GenerateSummaryText(document)
  local lines = stringx.splitlines(document, false)
  local summary = self:GenerateSummary(lines)
  local summary_lines = {}
  for i = 1, #summary do
    summary_lines[i] = stringx.join(' ', summary[i])
  end
  return stringx.join('\n', summary_lines)
end

function Summarizer:IsSkip(summary, sentence)
  local count = 0
  local existing = {}
  for i = 1, #summary do
    count = count + #summary[i]
    for j = 1, #summary[i] do
      existing[summary[i][j]] = 1
    end
  end
  if count > self.limit_ then
    return true
  end
  local new = 0
  for i = 1, #sentence do
    if existing[sentence[i]] == nil then
      new = new + 1
    end
  end
  if new * 2 > count then
    return false
  else
    return true
  end
end

function Summarizer:_GetTokenizedSentences(sentences)
  local tokenized = {}
  for i = 1, #sentences do
    local tokens = Utils.split(sentences[i])
    tokenized[i] = tokens
  end
  return tokenized
end

function Summarizer:_GetSentenceFeatures(sentences)
  local features = {}
  for i = 1, #sentences do
    local token_features = self:_GetTokenFeatures(sentences[i])
    local doc_features = self:_GetDocumentFeatures(sentences[i], i - 1)
    features[i] = {token_features, doc_features}
  end
  return features
end

function Summarizer:_GetSentenceScores(features)
  local scores = torch.FloatTensor(#features)
  for i = 1, #features do
    local s = self.model_:evaluate(features[i])
    scores[i] = s[1][1]
  end
  return scores
end

function Summarizer:_GetTokenFeatures(sentence)
    local tokens = sentence
    local token_features = torch.IntTensor(1, #tokens)
    for k = 1, #tokens do
      assert(tokens[k] ~= nil)
      token_features[1][k] = self.model_.word_vocab_:word_id(tokens[k])
    end
    return token_features
end

function Summarizer:_GetDocumentFeatures(sentence, pos)
    local tokens = sentence
    local avg_count = DataSet._GetAverageWordCount(tokens, self.model_.word_vocab_)
    local avg_category_count = DataSet._GetAverageCategoryCount(tokens, self.model_.word_vocab_)
    local document_features = torch.FloatTensor(1, 3)
    document_features[1][1] = pos
    document_features[1][2] = avg_count
    document_features[1][3] = avg_category_count
    return document_features
end
