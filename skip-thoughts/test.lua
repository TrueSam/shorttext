require("torch")
require("paths")
require("pl.stringx")
require("Model")
require("Config")

local function OutputVector(model, infile, outfile)
  local lines = io.lines(infile)
  local output = {}
  for i = 1, #lines do
    local v = model:vector(lines[i])
    local l = {}
    for k = 1, v:size(1) do
      table.insert(l, tostring(v[k]))
    end
    l = (' '):join(l)
    table.insert(output, l)
  end
  local file = io.open (outfile, 'w')
  local o = ('\n'):join(output)
  file:write(o)
end

local model = torch.load()
local config = Config()

local pos_infile = paths.concat(config.kDataPath, "rt.pos.wrd")
local pos_outfile = paths.concat(config.kDataPath, "rt.pos.vec")
local neg_infile = paths.concat(config.kDataPath, "rt.neg.wrd")
local neg_outfile = paths.concat(config.kDataPath, "rt.neg.vec")

OutputVector(model, pos_infile, pos_outfile, 1)
OutputVector(model, neg_infile, neg_outfile, -1)
