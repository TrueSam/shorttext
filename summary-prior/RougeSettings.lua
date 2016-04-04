require('torch')
require('pl')

local RougeSettings = torch.class('RougeSettings')

function RougeSettings.OutputSettingsXML(peer_files, model_dir, xml_file)
  local model_files = dir.getallfiles(model_dir)
  local align = RougeSettings._AlignPeerAndModelFiles(peer_files, model_files)
  RougeSettings._GenerateXMLSettings(align, xml_file)
end

function RougeSettings._AlignPeerAndModelFiles(peer_files, model_files)
  local align = {}
  for i = 1, #peer_files do
    local matched = nil
    for j = 1, #model_files do
      if stringx.endswith(path.basename(model_files[j]), path.basename(peer_files[i])) then
        matched = model_files[j]
      end
    end
    if matched ~= nil then
      if align[peer_files[i]] ~= nil then
        table.insert(align[peer_files[i]], matched)
      else
        align[peer_files[i]] = {matched}
      end
    end
  end
  return align
end

function RougeSettings._GenerateXMLSettings(aligned_peer_model_files, output_file)
  local eval_id = 0
  local rouge_lines = {}
  table.insert(rouge_lines, '<ROUGE_EVAL version="1.55">')
  for k, v in pairs(aligned_peer_model_files) do
    assert(#v > 0)
    table.insert(rouge_lines, '<EVAL ID="' .. eval_id .. '">')
    table.insert(rouge_lines, '  <MODEL-ROOT>' .. path.dirname(v[1]) .. '</MODEL-ROOT>')
    table.insert(rouge_lines, '  <PEER-ROOT>' .. path.dirname(k) .. '</PEER-ROOT>')
    table.insert(rouge_lines, '  <INPUT-FORMAT TYPE="SPL">  </INPUT-FORMAT>')
    table.insert(rouge_lines, '  <PEERS>')
    table.insert(rouge_lines, '    <P ID="0">' .. path.basename(k) .. '</P>')
    table.insert(rouge_lines, '  </PEERS>')
    table.insert(rouge_lines, '  <MODELS>')
    for i = 1, #v do
      local b = path.basename(v[i])
      local parts = stringx.split(b, '.')
      table.insert(rouge_lines, '    <M ID="' .. parts[5] .. '">' .. b .. '</M>')
    end
    table.insert(rouge_lines, '  </MODELS>')
    table.insert(rouge_lines, '</EVAL>')
    eval_id = eval_id + 1
  end
  table.insert(rouge_lines, '</ROUGE_EVAL>')
  assert(#rouge_lines > 2)
  utils.writefile(output_file, stringx.join('\n', rouge_lines))
end
