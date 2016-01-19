require("torch")
require("nn")

local CriterionTable, parent = torch.class("CriterionTable", "nn.Criterion")

function CriterionTable:__init(criterion)
  parent.__init(self)
  self.criterion = criterion
end

function CriterionTable:updateOutput(input, target) 
  assert(#input > 0)
  assert(#input == #target)
  self.output = 0
  for i = 1, #input do
    self.output = self.output + self.criterion:updateOutput(input[i], target[i])
  end
  collectgarbage()
  return self.output
end

function CriterionTable:updateGradInput(input, target)
  assert(#input > 0)
  assert(#input == #target)
  self.gradInput = {}
  for i = 1, #input do
    self.criterion:updateGradInput(input[i], target[i])
    assert(type(self.criterion.gradInput) == 'table')
    local clone = {}
    for k = 1, #self.criterion.gradInput do
      table.insert(clone, self.criterion.gradInput[k]:clone())
    end
    table.insert(self.gradInput, clone)
  end
  collectgarbage()
  return self.gradInput
end 
