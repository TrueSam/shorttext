require('torch')
require('posix')
require('pl.path')
require('Summarizer')
require('RougeSettings')

local Validator = torch.class('Validator')

function Validator.Validate(model, limit, epoch)
  assert(model ~= nil)
  assert(type(limit) == 'number')
  assert(type(epoch) == 'number')
  local summarizer = Summarizer(model, limit)
  local model_dir = 'data/duc2004/eval/models/1'
  local test_files = posix.glob.glob('./data/duc2004/docs/*/*')
  local validation_files = {
    'APW19981001.0299',
    'APW19981010.0164',
    'APW19981020.0241',
    'APW19981101.0843',
    'APW19981110.0240',
    'APW19981120.0274',
    'APW19981202.0880',
    'APW19981210.0305',
    'APW19981220.0356',
    'NYT19981001.0363',
    'NYT19981010.0022',
    'NYT19981020.0380',
    'NYT19981102.0465',
  }

  local validate_dir = 'data/build/validate/'
  local peer_files = {}
  for i = 1, #test_files do
    local b = path.basename(test_files[i])
    for j = 1, #validation_files do
      if b == validation_files[j] then
        local s = summarizer:GenerateSummaryFromFile(test_files[i])
        local f = path.join(validate_dir, validation_files[j])
        utils.writefile(f, s);
        table.insert(peer_files, f)
        break
      end
    end
  end

  local xml_file = path.join(validate_dir, 'settings.xml')
  local model_files = posix.glob.glob(model_dir + '/*/*')
  RougeSettings.OutputSettingsXML(peer_files, model_files, xml_file)
  local rouge_dir = 'rouge-1.5.5'
  local rouge_command = rouge_dir .. '/ROUGE-1.5.5.pl -e ' .. rouge_dir .. '/data -n 1 -n 2 -x -a -s ' .. xml_file
  local command = io.popen(rouge_command)
  local result = command:read('*a')
  command:close()
  print('Validation for epcoh ' .. epoch .. ':')
  print(result)
end
