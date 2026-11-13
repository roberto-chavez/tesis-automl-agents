import {pathToFileURL} from 'node:url';
const root='D:/Tesis/tesis-automl-agents';
const build=root+'/slides/.avance2-build';
const skill='C:/Users/rober/.codex/plugins/cache/openai-primary-runtime/presentations/26.1024.11018/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const res=await finalizePresentation({workspaceDir:root,candidatePath:build+'/candidate.pptx',finalPath:root+'/slides/Avance2_Tesis.pptx',pythonExecutable:'C:/Users/rober/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[3,5,11,12,14].flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:[3,5,11,12,14],requiredNativeChartOwnerSlides:[17],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Calibri']},verifyArtifactToolImport:true,receiptPath:root+'/.avance2-validation/validation.json'});
console.log(JSON.stringify(res));





