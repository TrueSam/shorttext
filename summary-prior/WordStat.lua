require('torch')

local WordStat = torch.class('WordStat')

function WordStat:__init()
  self.count = 0
  self.category_count = 0
end
