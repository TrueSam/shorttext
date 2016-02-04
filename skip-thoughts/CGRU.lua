-- conditional GRU

require("rnn")
require("nn")
require("torch")

local CGRU, parent = torch.class('CGRU', 'nn.GRU')


function CGRU:__init(inputSize, outputSize, rho)
   assert(inputSize > 0)
   assert(outputSize > 0)

   parent.__init(self, inputSize, outputSize, rho)
   self.inputs = {}
end

-------------------------- factory methods -----------------------------
function CGRU:buildModel()
   -- input : {{input, condition}, prevOutput}
   -- output : {output}
   local seq = nn.Sequential()

   -- Transform input to {input, prevOutput, condition}
   local concat = nn.ConcatTable()
   concat:add(nn.Sequential():add(nn.SelectTable(1)):add(nn.SelectTable(1)))
   concat:add(nn.SelectTable(2))
   concat:add(nn.Sequential():add(nn.SelectTable(1)):add(nn.SelectTable(2)))
   seq:add(concat)
   
   -- Calculate all four gates in one go : input, hidden, forget, output
   self.i2g = nn.Linear(self.inputSize, 2*self.outputSize)
   self.o2g = nn.LinearNoBias(self.outputSize, 2*self.outputSize)
   self.c2g = nn.LinearNoBias(self.outputSize, 2*self.outputSize)

   local para = nn.ParallelTable():add(self.i2g):add(self.o2g):add(self.c2g)
   local gates = nn.Sequential()
   gates:add(para)
   gates:add(nn.CAddTable())

   -- Reshape to (batch_size, n_gates, hid_size)
   -- Then slize the n_gates dimension, i.e dimension 2
   gates:add(nn.Reshape(2,self.outputSize))
   gates:add(nn.SplitTable(1,2))
   local transfer = nn.ParallelTable()
   transfer:add(nn.Sigmoid()):add(nn.Sigmoid())
   gates:add(transfer)

   local concat = nn.ConcatTable()
   concat:add(nn.Identity()):add(gates)
   seq:add(concat)
   seq:add(nn.FlattenTable()) -- x(t), s(t-1), c, r, z

   -- Rearrange to x(t), s(t-1), c, r, z, s(t-1)
   local concat = nn.ConcatTable()  -- 
   concat:add(nn.NarrowTable(1,5)):add(nn.SelectTable(2))
   seq:add(concat):add(nn.FlattenTable())

   -- h
   local hidden = nn.Sequential()
   local concat = nn.ConcatTable()
   local t1 = nn.Sequential()
   t1:add(nn.SelectTable(1)):add(nn.Linear(self.inputSize, self.outputSize))
   local t2 = nn.Sequential()
   t2:add(nn.NarrowTable(3,2)):add(nn.CMulTable()):add(nn.LinearNoBias(self.outputSize, self.outputSize))
   local t3 = nn.Sequential()
   t3:add(nn.SelectTable(3)):add(nn.LinearNoBias(self.outputSize, self.outputSize))
   concat:add(t1):add(t2):add(t3)
   hidden:add(concat):add(nn.CAddTable()):add(nn.Tanh())

   local z1 = nn.Sequential()
   z1:add(nn.SelectTable(5))
   z1:add(nn.SAdd(-1, true))  -- Scalar add & negation

   local z2 = nn.Sequential()
   z2:add(nn.NarrowTable(5,2))
   z2:add(nn.CMulTable())

   local o1 = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(hidden):add(z1)
   o1:add(concat):add(nn.CMulTable())

   local o2 = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(o1):add(z2)
   o2:add(concat):add(nn.CAddTable())

   seq:add(o2)

   return seq
end

function CGRU:updateOutput(input)
   local prevOutput
   if self.step == 1 then
      prevOutput = self.userPrevOutput or self.zeroTensor
      self.zeroTensor:resize(self.outputSize):zero()
      assert(#input == 2)  -- input and condition.
      if input[1]:dim() == 2 then
        self.zeroTensor:resize(input[1]:size(1), self.outputSize):zero()
      else
        self.zeroTensor:resize(self.outputSize):zero()
      end
   else
      -- previous output and cell of this module
      prevOutput = self.output
   end

   -- output(t) = gru{input(t), output(t-1)}
   local output
   if self.train ~= false then
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      -- the actual forward propagation
      output = recurrentModule:updateOutput{input, prevOutput}
   else
      output = self.recurrentModule:updateOutput{input, prevOutput}
   end

   if self.train ~= false then
     local input_ = self.inputs[self.step]
     self.inputs[self.step] = self.copyInputs and
     nn.rnn.recursiveCopy(input_, input) or
     nn.rnn.recursiveSet(input_, input)
   end

   self.outputs[self.step] = output

   self.output = output

   self.step = self.step + 1
   self.gradPrevOutput = nil
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   self.gradParametersAccumulated = false
   -- note that we don't return the cell, just the output
   return self.output
end
