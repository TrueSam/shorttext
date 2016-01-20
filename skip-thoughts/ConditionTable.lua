require('nn')
require('torch')

local ConditionTable = torch.class('ConditionTable', 'nn.Module')

function ConditionTable:__init(config)
  nn.Module.__init(self)
  self.config_ = config
  if config.useGPU == true  then
    require('cutorch')
  end
end

function ConditionTable:updateOutput(input)
  assert(type(input) == 'table')
  assert(#input == 2)
  assert(type(input[1]) == 'table')
  assert(#input[1] > 0)
  self.output = {}
  for i = 1, #input[1] do
    table.insert(self.output, {input[1][i], input[2]})
  end
  return self.output
end

function ConditionTable:updateGradInput(input, gradOutput)
  assert(#gradOutput == #self.output)
  for i = 1, #gradOutput do
    assert(#gradOutput[i] == 2)
    assert(torch.eq(torch.Tensor(gradOutput[i][2]:size()), torch.Tensor(input[2]:size())))
  end
  local gradInputs = {}
  local gradInput
  if self.config_.useGPU == true  then
    gradInput = torch.CudaTensor():resizeAs(gradOutput[1][2]):zero()
  else
    gradInput = torch.Tensor():resizeAs(gradOutput[1][2]):zero()
  end
  for i = 1, #gradOutput do
    gradInput:add(gradOutput[i][2])
    table.insert(gradInputs, gradOutput[i][1])
  end
  self.gradInput = {gradInputs, gradInput}
  return self.gradInput
end
