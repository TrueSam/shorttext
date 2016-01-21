require('torch')

local ValidationDataSet = torch.class('ValidationDataSet')

function ValidationDataSet:__init()
end

function ValidationDataSet:Precision(nearest)
end

function ValidationDataSet:Examples(nearest)
end

function ValidationDataSet.Show(examples, size, texts, sentences)
  assert(examples ~= nil)
  assert(size ~= nil)
  assert(texts ~= nil)
  assert(sentences ~= nil)
  assert(type(examples) == 'table')
  assert(#texts == size)
  assert(#sentences == size)

  local n = #examples

  for i = 1, n do
    local e = examples[i]
    local label = e[1]
    assert(label > 0 or label < 0)
    local base = e[2]
    local expt = e[3]
    assert(base ~= nil and base >= 1 and base <= size)
    assert(expt ~= nil and expt >= 1 and expt <= size)
    print('+:', texts[base])
    print('|:', sentences[base])
    if label > 0 then
      print('O')
    else
      print('X')
    end
    print('|:', texts[expt])
    print('+:', sentences[expt])
  end
end

