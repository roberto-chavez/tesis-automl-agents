import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const root='D:/Tesis/tesis-automl-agents/slides';
const p=await PresentationFile.importPptx(await FileBlob.load(root+'/Avance2_Tesis.pptx'));
for(let i=0;i<p.slides.items.length;i++){const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(root+`/.avance2-build/final-${String(i+1).padStart(2,'0')}.png`,new Uint8Array(await b.arrayBuffer()));}
console.log('Rendered '+p.slides.items.length+' final slides');
