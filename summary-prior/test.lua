require("torch")
require("posix")
require("paths")
require("pl.stringx")
require("pl.path")
require("pl.file")
require("pl.dir")
require("Model")
require("Summarizer")
require("Config")
require("Vocabulary")

local model = torch.load('./model-5000.t7')
local limit = 100
local summarizer = Summarizer(model, limit)
local files = dir.getfiles('./data/test/')
print(files)
for _, filename in pairs(files) do
  local abstract = summarizer:GenerateSummaryFromFile(filename)
  local peer_filename = path.basename(filename)
  local f = path.join('/tmp/peers/', 'S.' .. limit .. '.' .. peer_filename)
  file.write(f, abstract)
end


data/test/APW19981001.0299
data/test/APW19981001.0312
data/test/APW19981001.0315
data/test/APW19981001.0539
data/test/APW19981001.1177
data/test/APW19981002.0522
data/test/APW19981002.0550
data/test/APW19981002.0557
data/test/APW19981002.0567
data/test/APW19981002.0778
data/test/APW19981002.0783
data/test/APW19981002.0809
data/test/APW19981002.1081
data/test/APW19981003.0144
data/test/APW19981003.0292
data/test/APW19981003.0517
