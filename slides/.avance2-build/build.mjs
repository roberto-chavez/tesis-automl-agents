import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';
import sharp from 'sharp';

const root='D:/Tesis/tesis-automl-agents';
const build=root+'/slides/.avance2-build';
const skill='C:/Users/rober/.codex/plugins/cache/openai-primary-runtime/presentations/26.1024.11018/skills/presentations';
const run='runs/20261102-133428-608f';
const C={navy:'#0F2A4A',teal:'#1C7293',amber:'#F2A541',muted:'#52677C',paper:'#F7F9FB',white:'#FFFFFF'};
const p=Presentation.create({slideSize:{width:1280,height:720}});
const sources={thesis:'docs/20203540_RobertoChavez_PabloFonseca_E2.md, secciones 1.1 y 1.2',impl:'README.md; docs/orquestador_adaptativo.md; docs/especialistas.md',run:run+'/reports/06_reporte_final.md; '+run+'/final_evaluation.json'};
function text(s,t,x,y,w,h,size=28,color=C.navy,bold=false){const q=s.shapes.add({name:t.slice(0,45),geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});q.text=t;q.text.style={typeface:'Calibri',fontSize:size,color,bold};return q;}
function slide(title,notes,source,dark=false){const s=p.slides.add();s.background.fill=dark?C.navy:C.paper;if(title)text(s,title,64,47,1152,103,44,dark?C.white:C.navy,true);const n=p.slides.items.length;text(s,String(n).padStart(2,'0'),1172,668,50,28,17,dark?'#9FB3C8':C.muted);s.speakerNotes.text=notes+'\n\nFuentes del proyecto: '+source;return s;}
function para(s,head,body,x,y,w=525){text(s,head,x,y,w,47,29,C.teal,true);text(s,body,x,y+53,w,145,27);}
function footer(s,t){text(s,t,64,620,1120,42,20,C.muted);}
function table(s,values,widths,y=176,h=380,size=25){const tb=s.tables.add({rows:values.length,columns:values[0].length,left:64,top:y,width:1152,height:h,columnWidths:widths,values});tb.borders.assign({fill:'#D8E1E8',width:1});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){const cell=tb.getCell(r,c);cell.fill=r===0?C.navy:(r%2?C.white:'#EDF2F6');cell.text.style={typeface:'Calibri',fontSize:size,color:r===0?C.white:C.navy,bold:r===0};}return tb;}
async function screenshot(s,name,x,y,w,h,crop){let buf=await fs.readFile(build+'/'+name+'-full.jpg');if(crop)buf=await sharp(buf).extract(crop).jpeg({quality:95}).toBuffer();s.images.add({blob:buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength),contentType:'image/jpeg',alt:'Captura real de la aplicación AutoML: '+name,fit:'contain',position:{left:x,top:y,width:w,height:h}});}

let s=slide('', 'Presentación sugerida: 15 a 18 minutos para las diapositivas 1 a 16. Los anexos sirven para preguntas. Presentar el problema y el objetivo, explicar el prototipo y cerrar con la evaluación pendiente. El título abrevia el tema completo: Arquitectura multiagente con RAG para automatizar la construcción, selección, entrenamiento y preparación para el despliegue de modelos de clasificación de imágenes en pymes.',sources.thesis,true);
text(s,'SEGUNDO AVANCE DE TESIS',64,63,1100,40,24,C.amber,true);
text(s,'Arquitectura multiagente\ncon RAG para AutoML',64,155,1120,172,62,C.white,true);
text(s,'Clasificación de imágenes para pymes\nPrototipo, resultados preliminares y evaluación',68,354,1100,100,30,'#CCD9E6');
text(s,'Roberto Abel Chávez Javier',68,514,1080,42,29,C.white,true);
text(s,'Asesor: Pablo Alejandro Fonseca Arroyo',68,562,1100,40,24,'#CCD9E6');
text(s,'PUCP · Facultad de Ciencias e Ingeniería\n3 de noviembre de 2026',68,625,1000,62,20,'#9FB3C8');

s=slide('Problema y alcance','Explicar el escenario de una pyme que ya tiene imágenes etiquetadas, pero carece de un equipo especializado. Una aplicación posible es clasificar productos correctos y defectuosos, sin afirmar que el prototipo ya se validó en ese dominio. El sistema actual trabaja con clasificación supervisada, no con detección ni segmentación.',sources.thesis);
text(s,'Construir un clasificador exige decisiones\ntécnicas y ciclos de prueba y error.',64,164,1120,110,38,C.navy,true);
para(s,'Barreras de la pyme','Selección de arquitectura e hiperparámetros.\nFalta de especialistas y presupuesto de cómputo limitado.',64,316);
para(s,'Alcance de esta tesis','Dataset etiquetado por carpetas.\nModelo de clasificación, reporte comprensible y API preparada para integrar.',664,316);
footer(s,'Población objetivo: pymes cuyo negocio principal no es la investigación en IA o ML.');

s=slide('Objetivo general y objetivos específicos','El objetivo general mantiene el alcance del documento de tesis. Los cuatro objetivos se corresponden con las cuatro causas del problema. Aclarar que los indicadores de usabilidad y ahorro comparativo necesitan validación experimental posterior.',sources.thesis);
text(s,'Automatizar la construcción, selección, entrenamiento\ny preparación para el despliegue de clasificadores de imágenes.',64,156,1152,100,31,C.navy,true);
table(s,[['Objetivo','Resultado buscado'],['O1 · Acotar la búsqueda','RAG y configuraciones candidatas justificadas'],['O2 · Automatizar el ciclo','Entrenamiento, evaluación y ajustes sin supervisión continua'],['O3 · Facilitar el uso','Carga de datos y reporte para usuarios no expertos'],['O4 · Controlar el cómputo','Descarte temprano y medición de recursos']],[360,792],290,295,25);

s=slide('Avance desde la primera entrega','En el primer avance se presentaron los capítulos 1 a 3 y se propuso como siguiente hito el primer prototipo. Este segundo avance muestra una implementación integrada. No presentar la implementación como cumplimiento ya demostrado de todos los objetivos de investigación. La web actual usa HTML, CSS y JavaScript servidos por FastAPI, en lugar de React/Node.js planteado antes.','slides/Avance1_Tesis.pptx, diapositivas 8 y 14; README.md; docs/integracion_completa.md');
para(s,'Avance 1','Problema y cuatro objetivos.\nMarcos de referencia.\nRevisión de 14 estudios primarios.',64,185);
para(s,'Avance 2','Prototipo integrado y persistente.\nInterfaz de carga, seguimiento y resultados.\nEjecuciones y artefactos verificables.',664,185);
text(s,'La siguiente etapa es medir eficacia, costo y usabilidad\ncon un protocolo experimental común.',64,468,1110,105,35,C.teal,true);

s=slide('Arquitectura de la solución','Recorrer el flujo en orden: el usuario aporta el ZIP y sus restricciones. El perfil y el RAG preparan el contexto. El orquestador delega propuestas y diagnósticos, y ejecuta herramientas validadas. El motor entrena secuencialmente. Al cerrar se congela el ganador antes de abrir test. Los especialistas no son procesos autónomos que entrenan en paralelo.',sources.impl);
table(s,[['Componente','Responsabilidad'],['Entrada y perfil','Valida imágenes, fija particiones y resume entrenamiento'],['RAG y configuración','Recupera guías y propone arquitecturas e hiperparámetros'],['Orquestador y diagnóstico','Deciden continuar, ajustar, descartar o cerrar'],['Motor de ejecución','TensorFlow/Keras, Optuna TPE y poda Hyperband'],['Evaluación y entrega','Prueba final, reportes y API FastAPI con Dockerfile']],[350,802],164,395,24);
footer(s,'LangGraph coordina el flujo. SQLite conserva decisiones, intentos y presupuesto.');

s=slide('RAG y acotación de candidatas','RAG significa generación aumentada por recuperación. La base inicial contiene 23 documentos redactados a mano: 11 fichas de arquitecturas y 12 guías. La ejecución utiliza embeddings hash, no embeddings semánticos. Recuperó 17 documentos. La reducción mostrada incluye catálogo, filtros de recursos y propuesta del especialista; no debe atribuirse causalmente sólo al RAG. Cada propuesta se valida contra el catálogo y la rejilla. Si el proveedor falla, se usan reglas.','README.md; docs/rag_y_contexto.md; '+run+'/config.json; '+run+'/reports/01_criterios_acotacion.json');
text(s,'990',64,164,440,118,92,C.navy,true);text(s,'configuraciones del catálogo',70,282,475,40,26);
text(s,'16',680,164,440,118,92,C.teal,true);text(s,'configuraciones candidatas',686,282,480,40,26);
para(s,'Contexto para decidir','Perfil de entrenamiento, recursos disponibles y guías recuperadas.',64,376);
para(s,'Ejemplo del 2 de noviembre','EfficientNetV2B0 y MobileNetV2.\nReducción de espacio: 98,4 %.',664,376);
footer(s,'Acotar el espacio no equivale a demostrar un ahorro de tiempo del 98,4 %.');

s=slide('Búsqueda adaptativa y evaluación separada','Explicar que el LLM recomienda y el código valida. El orquestador observa resultados y presupuesto después de bloques secuenciales. La selección usa validación y desempata por menor tamaño del catálogo. El cierre congela el checkpoint y bloquea nuevas operaciones de búsqueda. Los especialistas no acceden al test. La reanudación conserva intentos confirmados, pero reinicia desde cero el entrenamiento de un intento interrumpido.','docs/orquestador_adaptativo.md; docs/especialistas.md; README.md');
para(s,'Durante la búsqueda','El orquestador consulta resultados, solicita diagnósticos y revisa candidatas dentro del presupuesto.',64,177);
para(s,'Al terminar la búsqueda','El sistema congela el modelo elegido con validación y lo evalúa una vez en el conjunto de prueba.',664,177);
text(s,'Controles implementados',64,440,1100,44,31,C.teal,true);
text(s,'Particiones persistentes y control de duplicados.\nLímites de tiempo, intentos y llamadas LLM.\nRegistro de decisiones y recuperación tras interrupciones.',64,500,1100,114,27);

s=slide('Interfaz: preparación del proyecto','El usuario carga un ZIP con una carpeta por clase y describe qué quiere distinguir. La interfaz ofrece presupuestos de búsqueda. Los tiempos de preparación y cierre no forman parte del límite y una época puede terminar después del plazo. Captura real del prototipo local, tomada para esta presentación.','http://127.0.0.1:8000/; docs/web_y_jobs.md');
await screenshot(s,'inicio',408,148,808,465,{left:24,top:550,width:1205,height:900});
text(s,'ZIP por clases',64,183,320,50,31,C.teal,true);text(s,'Imágenes etiquetadas y legibles.',64,245,300,90,27);
text(s,'Presupuesto',64,374,320,50,31,C.teal,true);text(s,'El usuario define cuánto explorar.',64,436,300,100,27);
footer(s,'La aplicación funciona localmente y permite una ejecución de entrenamiento a la vez.');

s=slide('Interfaz: seguimiento de la ejecución','Esta captura corresponde a una ejecución terminada, no a una simulación de progreso. El seguimiento muestra estados e intentos reales. La ejecución agotó el plazo de búsqueda, con ocho intentos y mejor validación de 88,4 %. El plazo cooperativo explica el pequeño exceso sobre diez minutos. No confundir intentos con modelos completamente entrenados.','http://127.0.0.1:8000/#tracking/20261102-133428-608f; '+run+'/reports/05_benchmark.json');
await screenshot(s,'seguimiento',405,150,810,465,{left:24,top:300,width:1205,height:695});
text(s,'8 intentos',64,187,315,50,34,C.teal,true);text(s,'5 completos\n2 podados\n1 cancelado',64,255,300,130,28);
text(s,'Decisiones trazables',64,433,310,90,30,C.teal,true);text(s,'Eventos y presupuesto conservados.',64,524,310,70,26);

s=slide('Interfaz: resultados y entrega','El usuario recibe el modelo, los reportes y la API cuando las comprobaciones locales lo permiten. La captura distingue calidad y verificación. En esta ejecución el objetivo del 99 % no se alcanzó y Docker no se comprobó, aunque las pruebas locales de API aprobaron. La comprobación Docker de otros artefactos no cambia el estado de este experimento.','http://127.0.0.1:8000/#results/20261102-133428-608f; '+run+'/reports/07_entrega.json');
await screenshot(s,'resultados',405,150,810,465,{left:24,top:462,width:1205,height:695});
text(s,'MobileNetV2',64,180,320,50,34,C.teal,true);text(s,'Modelo y métricas de prueba disponibles.',64,250,305,100,27);
text(s,'Entrega explícita',64,406,320,50,30,C.teal,true);text(s,'Calidad, archivos y verificación se informan por separado.',64,471,310,130,27);

s=slide('Resultado preliminar: clasificación de dígitos','Caso: imágenes de dígitos de 0 a 9. El repositorio registra 1000 imágenes y particiones de 708, 146 y 146. No se afirma una procedencia pública específica porque la evidencia inspeccionada no la identifica. La búsqueda duró aproximadamente diez minutos, con transferencia ImageNet y hasta cinco épocas por intento. La exactitud de prueba corresponde a 127 aciertos de 146. Una sola ejecución no estima estabilidad ni demuestra eficacia en pymes.',sources.run+'; '+run+'/data/split_manifest.json; '+run+'/config.json');
text(s,'87,0 %',64,159,520,115,84,C.teal,true);text(s,'exactitud en prueba',70,282,500,46,29);text(s,'F1 macro: 0,869\n127 aciertos de 146 imágenes',70,351,510,106,29);
table(s,[['Partición','Imágenes'],['Entrenamiento','708'],['Validación','146'],['Prueba','146']],[300,240],182,300,27).position={left:660,top:182,width:540,height:300};
text(s,'Validación: 88,4 %\nObjetivo configurado: 99 %',664,510,530,90,28);
footer(s,'IC 95 % aproximado de la exactitud de prueba: 80,6–91,5 %. Resultado de una ejecución.');

s=slide('Eficiencia: qué está medido y qué falta','Se registraron 8 intentos: 5 completos, 2 podados y 1 cancelado. El cálculo de ahorro de épocas compara 29 épocas de siete intentos elegibles con 35 épocas máximas para esos mismos intentos: 17,1 %. Excluye el intento cancelado; el total general es 32 épocas. Las líneas base exhaustivas sólo son estimaciones basadas en costo medio, no ejecuciones reales. No reportar 24,1 horas como tiempo efectivamente ahorrado.',''+run+'/reports/05_benchmark.json');
table(s,[['Evidencia de esta ejecución','Interpretación'],['990 a 16 configuraciones','98,4 % de reducción del espacio inicial'],['2 de 8 intentos podados','El sistema descarta alternativas durante el entrenamiento'],['29 de 35 épocas elegibles','17,1 % menos épocas en esos siete intentos'],['Línea base exhaustiva estimada','Falta una comparación real con presupuesto común']],[530,622],175,380,25);
footer(s,'La reducción de épocas no mide por sí sola el ahorro de tiempo, energía o costo.');

s=slide('Verificación técnica disponible','La evidencia histórica de integración del 24 de octubre registra 293 pruebas aprobadas, incluida una integración real con Gemini. No es una suite ejecutada de nuevo para esta presentación. Los recorridos de recuperación interrumpen procesos de prueba y verifican la conservación de trabajo confirmado. Docker se comprobó en cinco artefactos de integración. Son imágenes sintéticas para validar integración, no un benchmark de calidad predictiva.','docs/integracion_completa.md; runs/stage11/docker-20261024/verification.json; runs/stage11/verification.json');
text(s,'293',64,171,420,120,88,C.teal,true);text(s,'pruebas aprobadas\nen la verificación del 24/10',70,296,450,95,29);
text(s,'Evidencia de integración',654,181,540,45,32,C.navy,true);
text(s,'Recorrido web y CLI.\nCancelación y recuperación tras caídas.\nIdentidad del modelo seleccionado.\nAPI local y contenedor Docker.',654,254,545,234,28);
footer(s,'Las pruebas de integración comprueban funcionamiento. La evaluación científica sigue pendiente.');

s=slide('Evaluación experimental propuesta','Presentar como trabajo próximo, no como experimentos ya realizados. Definir un dataset de un dominio de pyme, repetir semillas, fijar particiones y el mismo equipo. Comparar la arquitectura completa frente a reglas y una búsqueda sin acotación. Las ablaciones separarán el aporte de RAG y del orquestador. Para usuarios medir tareas de carga e interpretación, SUS y comprensión del reporte. Acordar con el asesor número de datasets, semillas, usuarios y presupuestos.','docs/20203540_RobertoChavez_PabloFonseca_E2.md, secciones 1.2.4 y 1.3; README.md, próximos pasos');
table(s,[['Pregunta','Diseño propuesto'],['¿Mejora la calidad por presupuesto?','Líneas base reales, mismos datos, equipo y tiempo'],['¿Qué aporta cada componente?','Ablaciones de RAG y de decisiones LLM'],['¿Es estable el resultado?','Varias semillas y datasets, con dispersión de métricas'],['¿Puede usarlo una persona no experta?','Tareas observadas, SUS y comprensión del reporte']],[510,642],175,380,25);
footer(s,'Métricas: exactitud, F1 macro, tiempo, memoria, llamadas LLM y tasa de tareas completadas.');

s=slide('Limitaciones y próximos entregables','Distinguir limitaciones del sistema y pendientes de investigación. Hay catálogo cerrado, entrenamiento de cabeza con backbone congelado y una base de conocimiento manual. El prototipo es local, sin autenticación. La aplicación a pymes y la usabilidad requieren evidencia externa. Proponer secuencia de trabajo, no fechas ni compromisos no acordados.','README.md; docs/03_hoja_de_ruta_y_pasos.md; docs/20203540_RobertoChavez_PabloFonseca_E2.md');
para(s,'Límites actuales','Catálogo de 11 arquitecturas.\nSin fine-tuning del backbone.\nBase RAG manual de 23 documentos.\nSin estudio con usuarios objetivo.',64,170);
para(s,'Próximos entregables','Protocolo y datasets de evaluación.\nComparación con líneas base reales.\nPruebas de usabilidad.\nCapítulos de resultados y conclusiones.',664,170);
text(s,'Prioridad: convertir el prototipo funcional\nen evidencia reproducible para la tesis.',64,502,1130,100,36,C.teal,true);

s=slide('Conclusiones del segundo avance','Cerrar retomando el objetivo general. Existe un prototipo que integra la carga de imágenes, la búsqueda controlada y la entrega. La ejecución preliminar ilustra el flujo y sus límites. Falta comprobar la contribución frente a alternativas, la estabilidad y el uso por no expertos. Invitar preguntas sobre arquitectura, evaluación y alcance.',sources.impl+'; '+sources.run,true);
text(s,'El prototipo integra la búsqueda,\nla evaluación y la entrega del modelo.',64,174,1120,112,42,C.white,true);
text(s,'La ejecución con dígitos alcanzó 87,0 % de exactitud en prueba.\nLa comparación experimental y la validación con usuarios\ndefinirán el alcance del aporte.',64,354,1120,150,31,'#CCD9E6');
text(s,'Preguntas',64,565,1120,65,42,C.amber,true);

s=slide('Anexo: desempeño por clase','Usar sólo si preguntan dónde falla el modelo. F1 por clase calculado sobre el conjunto de prueba de 146 imágenes. Las clases 5 y 7 tienen los F1 menores. Con soportes de 12 a 18 imágenes por clase las conclusiones deben ser prudentes. Revisar errores e imágenes reales antes de ajustar el sistema.',sources.run);
const ev=JSON.parse(await fs.readFile(root+'/'+run+'/final_evaluation.json','utf8'));
const chart=s.charts.add('bar',{position:{left:64,top:174,width:1152,height:390},categories:Object.keys(ev.metrics.per_class),series:[{name:'F1 por clase',values:Object.values(ev.metrics.per_class).map(x=>x.f1),fill:C.teal}],hasLegend:false,barOptions:{direction:'column',gapWidth:80},yAxis:{min:0,max:1,majorUnit:0.2,numberFormatCode:'0.0',textStyle:{typeface:'Calibri',fontSize:20}},xAxis:{textStyle:{typeface:'Calibri',fontSize:23}},dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:'Calibri',fontSize:18},numberFormatCode:'0.00'},chartFill:C.paper,plotAreaFill:C.paper});
footer(s,'Dígitos del 0 al 9. Menores F1: clase 5 (0,750) y clase 7 (0,759).');

s=slide('Anexo: fuentes y reproducción','Todas las cifras del ejemplo proceden de una misma ejecución. Los registros de integración pertenecen a verificaciones previas identificadas por fecha. Las capturas proceden de la aplicación local suministrada. Consultar las fuentes en las notas de cada diapositiva. Para demostrar el prototipo sin iniciar un entrenamiento, abrir la URL de resultados del experimento.','Fuentes locales enumeradas en esta diapositiva y en las notas.');
text(s,'Documento de tesis y avance anterior',64,170,1100,45,29,C.teal,true);text(s,'Objetivos, alcance y estado de la primera entrega.',64,220,1100,43,26);
text(s,'Documentación de implementación',64,300,1100,45,29,C.teal,true);text(s,'README y documentos de orquestación, RAG, especialistas e integración.',64,350,1100,50,26);
text(s,'Ejecución de referencia',64,430,1100,45,29,C.teal,true);text(s,'20261102-133428-608f\nReportes de criterios, benchmark, evaluación y entrega.',64,480,1100,90,26);
footer(s,'Demo: http://127.0.0.1:8000/#results/20261102-133428-608f');

await fs.writeFile(build+'/deck.json',JSON.stringify(p.toProto()));
await (await PresentationFile.exportPptx(p)).save(build+'/candidate.pptx');
for(let i=0;i<p.slides.items.length;i++){let b=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(build+`/slide-${String(i+1).padStart(2,'0')}.png`,new Uint8Array(await b.arrayBuffer()));}
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const res=await finalizePresentation({workspaceDir:root,candidatePath:build+'/candidate.pptx',finalPath:root+'/slides/Avance2_Tesis.pptx',pythonExecutable:'C:/Users/rober/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[3,5,11,12,14].flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:[3,5,11,12,14],requiredNativeChartOwnerSlides:[17],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Calibri']},verifyArtifactToolImport:true,receiptPath:root+'/.avance2-validation/validation.json'});
console.log(JSON.stringify(res));





