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

local model_file = './data/build/model-2000.t7'
local model = torch.load(model_file)
local limit = 100
local summarizer = Summarizer(model, limit)
local docs_dir = './data/duc2004/docs/*/*'
local doc_files = posix.glob.glob(docs_dir)
local peer_files = {}
for _, filename in pairs(doc_files) do
  local abstract = summarizer:GenerateSummaryFromFile(filename)
  local peer_filename = path.basename(filename)
  local f = path.join('./data/build/peers/', 'S.' .. limit .. '.' .. peer_filename)
  file.write(f, abstract)
  table.insert(peer_files, f)
end

local summary_model_dir = 'data/duc2004/eval/models/1'
local summary_model_files = posix.glob.glob(summary_model_dir .. '/*')
local test_dir = './data/test/'
local xml_file = path.join(test_dir, 'settings.xml')
RougeSettings.OutputSettingsXML(peer_files, summary_model_files, xml_file)
local rouge_dir = 'rouge-1.5.5'
local rouge_command = rouge_dir .. '/ROUGE-1.5.5.pl -e ' .. rouge_dir .. '/data -n 1 -n 2 -x -a -s ' .. xml_file
local command = io.popen(rouge_command)
local result = command:read('*a')
command:close()
print('Testing for model ' .. model_file .. ':')
print(result)
