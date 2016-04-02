require("torch")
require("paths")
require("pl.stringx")
require("Model")
require("Summarizer")
require("Config")
require("Vocabulary")

local model = torch.load('./model-5.t7')

local parameters, gradParameters = model.model_:getParameters()

print(parameters:norm())
print(gradParameters:norm())

local test_filename = './data/test/APW19981029.0570'

local summarizer = Summarizer(model, 100)

local abstract = summarizer:GenerateSummaryFromFile(test_filename)

print(abstract)
