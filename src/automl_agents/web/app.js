"use strict";
const $ = id => document.getElementById(id);
const terminal = new Set(["completed", "target_not_reached", "no_model", "artifact_failed", "cancelled", "interrupted", "failed"]);
const labels = {queued:"En espera",running:"En ejecución",cancelling:"Cancelación solicitada",cancelled:"Cancelada",interrupted:"Interrumpida",failed:"Error",completed:"Finalizada",target_not_reached:"Finalizada · objetivo no alcanzado",no_model:"Sin modelo",artifact_failed:"Entrega parcial"};
const descriptions = {
  queued:"Su proyecto espera su turno. Sólo se entrena un proyecto a la vez.",
  running:"Puede salir de esta página y volver después. El servidor conserva su avance.",
  cancelling:"Solicitud guardada. Se detendrá al terminar la época o la operación en curso; puede tardar unos minutos.",
  cancelled:"El entrenamiento se detuvo. Puede reanudarlo con los mismos datos y el presupuesto restante. El límite de tiempo también cuenta las pausas.",
  interrupted:"El servidor se detuvo antes de terminar. Reanude para continuar desde los resultados guardados.",
  failed:"La ejecución encontró un error. Revise el mensaje y corrija la causa antes de reanudar.",
  completed:"El modelo alcanzó el objetivo de validación. Revise también sus resultados de prueba y la verificación de la API.",
  target_not_reached:"Hay un modelo disponible, aunque no alcanzó el objetivo de validación. Revise su calidad antes de usarlo.",
  no_model:"La búsqueda terminó sin un modelo elegible. Revise los reportes para preparar otra prueba.",
  artifact_failed:"La entrega quedó incompleta. Los resultados y reportes disponibles se conservan para su revisión."
};
const stageList = [["perfil","Revisar y preparar las imágenes"],["rag","Consultar buenas prácticas"],["candidatas","Elegir alternativas"],["entrenar","Entrenar y comparar"],["evaluar","Evaluar con imágenes de prueba"],["reportar","Preparar reportes"],["empaquetar","Empaquetar y comprobar la API"]];
const stageLabels = Object.fromEntries(stageList);
const reasons = {target_reached:"Se alcanzó el objetivo de validación.",budget_exhausted:"Se agotó un límite de búsqueda.",stagnation:"Las últimas pruebas no mejoraron el resultado.",no_progress:"No se encontraron nuevas acciones útiles.",user_cancelled:"Cancelación solicitada por usted.",cancelled:"Cancelación solicitada por usted.",search_time_exhausted:"Se agotó el tiempo de búsqueda.",training_attempts_exhausted:"Se agotó el límite de intentos.",no_model:"No se encontró un modelo elegible.",packaging_failed:"No se pudo completar el paquete de la API.",final_evaluation_failed:"Falló la evaluación final.",all_candidates_discarded:"Se descartaron las alternativas disponibles."};
let runId = null, data = null, view = "start", timer = null, failures = 0, actionPending = false, reportRun = null, pendingView = null;
const pct = n => Number.isFinite(n) ? `${(100*n).toLocaleString("es-PE", {maximumFractionDigits:1})} %` : "—";
const duration = n => Number.isFinite(n) ? (n < 60 ? `${Math.floor(n)} s` : `${Math.floor(n/60)} min ${Math.floor(n%60)} s`) : "—";
function text(id, value) { $(id).textContent = value; }
function remember(id) { try { localStorage.setItem("automl.run", id); } catch {} }
function remembered() { try { return localStorage.getItem("automl.run"); } catch { return null; } }
function showView(next, focus = true) {
  if (next !== "start" && !data) return;
  if (next === "results" && !terminal.has(data.status)) return;
  view = next;
  for (const name of ["start","tracking","results"]) $(name).hidden = name !== next;
  document.querySelectorAll("[data-view]").forEach(button => {
    if (button.dataset.view === next) button.setAttribute("aria-current","step");
    else button.removeAttribute("aria-current");
  });
  const hash = runId ? `#${next}/${runId}` : "#start";
  if (location.hash !== hash) history.replaceState(null,"",hash);
  if (focus) $(`${next}-title`).focus();
}
async function request(url, options = {}) {
  const response = await fetch(url, {signal:AbortSignal.timeout(30000), ...options});
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = body.detail;
    const error = new Error(typeof detail === "string" ? detail : detail?.message || "No se pudo completar la solicitud.");
    error.field = detail?.field; error.status = response.status; throw error;
  }
  return response;
}
function list(id, items, transform) {
  const children = items.map(item => { const li = document.createElement("li"); transform(li,item); return li; });
  $(id).replaceChildren(...children);
}
function drawChart(curves) {
  const series = [["accuracy","Entrenamiento","#174f42"],["val_accuracy","Validación","#80641f"]].filter(([key]) => curves?.[key]?.length);
  $("chart").replaceChildren();
  if (!series.length) { $("chart").textContent = "Las curvas aparecerán cuando termine el primer paso."; return; }
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns,"svg"); svg.setAttribute("viewBox","0 0 400 150"); svg.setAttribute("role","img");
  svg.setAttribute("aria-label","Acierto por época, entre cero y cien por ciento. " + series.map(([key,label]) => `${label}: ${curves[key].map(pct).join(", ")}`).join(". "));
  const n = Math.max(...series.map(([key]) => curves[key].length));
  for (const y of [0, .5, 1]) {
    const line = document.createElementNS(ns,"line");
    for (const [attr,v] of Object.entries({x1:32,x2:390,y1:130-y*115,y2:130-y*115,stroke:"#dce3dc"})) line.setAttribute(attr,v);
    svg.append(line);
    const label = document.createElementNS(ns,"text"); label.setAttribute("x",0); label.setAttribute("y",134-y*115); label.setAttribute("font-size",10); label.setAttribute("fill","#5b6a64"); label.textContent = `${y*100}%`; svg.append(label);
  }
  for (const [key,,color] of series) {
    const points = curves[key].map((y,x) => [32+x/Math.max(1,n-1)*350,130-Math.min(1,Math.max(0,y))*115]);
    const line = document.createElementNS(ns,"polyline"); line.setAttribute("points",points.map(p=>p.join(",")).join(" ")); line.setAttribute("fill","none"); line.setAttribute("stroke",color); line.setAttribute("stroke-width",2.5); svg.append(line);
    for (const [x,y] of points) { const dot=document.createElementNS(ns,"circle"); dot.setAttribute("cx",x); dot.setAttribute("cy",y); dot.setAttribute("r",3); dot.setAttribute("fill",color); svg.append(dot); }
  }
  const legend = document.createElement("div"); legend.className="legend";
  for (const [,label,color] of series) { const span=document.createElement("span"); span.style.color=color; span.textContent=`● ${label}`; legend.append(span); }
  $("chart").append(svg,legend);
}
function render(d) {
  const done = terminal.has(d.status);
  $("current").hidden=false; text("run-id",d.run_id); text("run-state",labels[d.status] || d.status);
  document.querySelector('[data-view="tracking"]').disabled=false;
  document.querySelector('[data-view="results"]').disabled=!done;
  text("tracking-title",done ? "Su ejecución ha terminado." : d.status === "queued" ? "Su proyecto está en espera." : "Estamos trabajando en sus imágenes.");
  text("status-description",descriptions[d.status] || "Consultando el estado de su proyecto.");
  const announcement = [labels[d.status],d.message].filter(Boolean).join(" · ");
  if ($("live-status").textContent !== announcement) text("live-status",announcement);
  $("live-status").classList.toggle("warning",["failed","interrupted","cancelling"].includes(d.status));
  const errorMessage=["failed","artifact_failed"].includes(d.status) ? d.summary?.error?.message || d.message : null;
  $("run-error").hidden=!errorMessage; text("run-error",errorMessage || "");
  $("cancel").hidden=!d.capabilities.cancel; $("resume").hidden=!d.capabilities.resume; $("see-results").hidden=!done;
  $("result-resume").hidden=!d.capabilities.resume;
  $("cancel").disabled=actionPending; $("resume").disabled=actionPending; $("result-resume").disabled=actionPending;
  text("stage",d.status === "queued" ? "En espera" : stageLabels[d.stage] || labels[d.status]);
  text("task",d.active_task?.description || (done ? "Consulte los resultados y archivos disponibles." : "Preparando la siguiente operación."));
  text("attempts",d.progress.attempts_finished); text("attempt-limit",`De un máximo de ${d.progress.attempts_limit} intentos, incluidos fallos y reintentos.`);
  text("time",duration(d.budget.consumed.search_seconds || 0)); text("time-limit",`Límite: ${duration(d.budget.limits.max_search_seconds)}. Se conserva al reanudar.`);
  const used=d.budget.consumed, limits=d.budget.limits;
  text("budget-detail",`Decisiones: ${used.decisions || 0} de ${limits.max_decisions}. Consultas al asistente: ${used.llm_calls || 0} de ${limits.max_llm_calls}. Reintentos: ${used.retries || 0} de ${limits.max_retries}. Estos límites se comparten entre todos los componentes.`);
  text("best",pct(d.best_validation?.accuracy));
  const current = d.progress.current_trial;
  text("epoch",current ? `${current.architecture} · ${current.epochs_completed} de hasta ${current.epochs_limit} pasos terminados.` : "No hay un intento activo. Los resultados confirmados permanecen guardados.");
  drawChart(current?.curves);
  // Las etapas completadas proceden de tiempos publicados por los nodos, no de un temporizador visual.
  const timings = d.execution.stage_timings || {};
  list("stages",stageList,(li,[key,label]) => { const stopped=key===d.stage && ["cancelled","interrupted","failed"].includes(d.status); const active = key===d.stage && !done; const finished=Object.hasOwn(timings,key) && !active && !stopped; li.className=active ? "active" : finished ? "done" : ""; li.textContent=`${label}${active ? " · en curso" : stopped ? " · detenida" : finished ? " · procesado" : ""}`; });
  list("plan",d.plan.tasks,(li,t)=>{li.textContent=`${t.description} · ${ {pending:"pendiente",running:"en curso",completed:"comprobado",failed:"falló",skipped:"omitido"}[t.status] || t.status}`;});
  list("decisions",d.decisions,(li,item)=>{li.textContent=item.reason;});
  list("events",[...d.events].reverse(),(li,event)=>{const time=document.createElement("time");time.textContent=new Date(event.ts*1000).toLocaleTimeString("es-PE");time.dateTime=new Date(event.ts*1000).toISOString();const message=document.createElement("span");message.textContent=event.message;li.append(time,message);});
  const s=d.summary || {}, winner=s.winner || d.execution.selection;
  text("result-description",descriptions[d.status] || "");
  text("winner",winner?.architecture || "Sin modelo seleccionado");
  text("test-accuracy",pct(s.test?.accuracy)); text("f1",Number.isFinite(s.test?.f1_macro) ? s.test.f1_macro.toFixed(3) : "—"); text("n-test",s.test?.n_test ?? "—"); text("val-accuracy",pct(winner?.val_acc));
  const target=winner?.metric_policy?.threshold;
  text("quality",winner && Number.isFinite(target) ? `${winner.val_acc >= target ? "Objetivo de validación alcanzado" : "Objetivo de validación no alcanzado"} (${pct(target)}).` : "No hay calidad final comprobada.");
  text("delivery-state",d.status === "artifact_failed" ? "Entrega parcial. Consulte las descargas disponibles." : Object.keys(d.downloads).length ? "Archivos conservados. Elija una descarga abajo." : "Sin archivos disponibles.");
  const verification=s.verification || d.execution.verification || {};
  const checks=verification.checks || {};
  text("verification",verification.status === "passed" ? "Comprobaciones aprobadas." : verification.status === "failed" ? "Falló una comprobación. La API no se ofrece como entrega válida." : checks.local?.status === "passed" ? "Pruebas locales aprobadas. Docker no comprobado; verificación completa pendiente." : "Verificación no ejecutada.");
  text("reason",reasons[d.stop_reason] || d.stop_reason || d.message || (d.status==="cancelled" ? "Cancelación solicitada por usted." : "No hay motivo registrado."));
  $("downloads").replaceChildren();
  const downloadLabels={all:"Todos los archivos disponibles",model:"Modelo (.keras)",reports:"Reportes (.zip)",artifact:"API para integrar (.zip)"};
  for (const kind of ["all","model","reports","artifact"]) if(d.downloads[kind]) { const a=document.createElement("a"); a.className="download";a.href=d.downloads[kind];a.textContent=downloadLabels[kind];$("downloads").append(a); }
  $("no-downloads").hidden=Boolean(Object.keys(d.downloads).length);
  $("report-details").hidden=!d.downloads.reports;
  if(reportRun!==d.run_id) { $("report-details").open=false; reportRun=null; text("report","Abra esta sección para cargar el reporte."); }
  const baseWarnings=["El resultado depende de la variedad y calidad de sus imágenes.","Compruebe el modelo con fotos reales de su caso de uso antes de integrarlo."];
  list("limitations",[...baseWarnings,...d.warnings],(li,w)=>{li.textContent=w;});
}
async function poll() {
  clearTimeout(timer);
  const id=runId;
  if (!id) return;
  try {
    const next=await (await request(`/api/runs/${id}`)).json();
    if(id!==runId) return;
    const changed=next.status!==data?.status;
    data=next; failures=0; $("connection").hidden=true; render(data);
    if(changed) refreshHistory();
    if(pendingView) {
      const requested=pendingView;pendingView=null;
      showView(requested.name==="results" && !terminal.has(data.status) ? "tracking" : requested.name,requested.focus);
    } else if(changed && terminal.has(data.status) && view==="tracking") showView("results");
    if(!terminal.has(data.status)) timer=setTimeout(poll,1800);
  } catch(error) {
    if(id!==runId) return;
    failures++;
    $("connection").hidden=false;
    text("connection",error.status===404 ? "No encontramos este proyecto en el servidor. Puede elegir otro en la lista o crear uno nuevo." : "No se pudo consultar el servidor. Su proyecto sigue guardado; reintentaremos la conexión automáticamente.");
    if(error.status!==404) timer=setTimeout(poll,Math.min(15000,2000*failures));
  }
}
async function openRun(id,next="tracking",focus=true) {
  if(!/^[0-9a-z-]{8,40}$/.test(id)) return;
  clearTimeout(timer); runId=id; data=null; remember(id); reportRun=null;pendingView={name:next,focus};
  for(const name of ["cancel","resume","result-resume"]) $(name).hidden=true;
  await poll();
}
async function refreshHistory() {
  try {
    const body=await (await request("/api/runs")).json();
    $("history-empty").hidden=body.runs.length>0;
    list("history-list",body.runs.slice(0,20),(li,run)=>{const button=document.createElement("button");const code=document.createElement("code");code.textContent=run.run_id;const badge=document.createElement("span");badge.className="badge";badge.textContent=labels[run.status] || run.status;button.append(code,badge);button.addEventListener("click",()=>openRun(run.run_id,terminal.has(run.status)?"results":"tracking"));li.append(button);});
  } catch { text("history-empty","No se pudo actualizar la lista. Use «Actualizar lista» para reintentar."); $("history-empty").hidden=false; }
}
let previewWorker = null, previewTimer = null, previewUrls = [];
function previewFile(file) {
  previewWorker?.terminate();
  previewWorker=null;
  clearTimeout(previewTimer);
  previewUrls.forEach(url=>URL.revokeObjectURL(url));
  previewUrls=[];
  $("file-previews").replaceChildren();
  $("file-previews").hidden=true;
  $("file-fallback").hidden=false;
  if(!file || !file.name.toLowerCase().endsWith(".zip") || file.size>512000000) return;
  try {
    const worker=new Worker("/assets/zip-preview.js?v=20261030-previews");
    previewWorker=worker;
    const stop=()=>{worker.terminate();if(previewWorker===worker) previewWorker=null;clearTimeout(previewTimer);};
    previewTimer=setTimeout(stop,15000);
    worker.onerror=stop;
    worker.onmessage=({data})=>{
      if(previewWorker!==worker) return;
      if(data.done) {stop();return;}
      const url=URL.createObjectURL(data.blob);
      previewUrls.push(url);
      const image=document.createElement("img");
      image.src=url;image.alt=`Vista previa de ${data.label}`;image.title=data.label;
      image.width=48;image.height=48;
      $("file-previews").append(image);
      $("file-previews").hidden=false;
      $("file-fallback").hidden=true;
    };
    worker.postMessage(file);
  } catch { /* The file remains selectable when previews are unavailable. */ }
}
function updateFileSelection() {
  const file=$("file").files[0];
  previewFile(file);
  $("file-picker").hidden=Boolean(file);
  $("file-card").hidden=!file;
  text("file-name",file ? file.name : "");
  let size="";
  if(file) {
    const unit=file.size>=1000000 ? "MB" : file.size>=1000 ? "KB" : "B";
    const divisor=unit==="MB" ? 1000000 : unit==="KB" ? 1000 : 1;
    size=`${(file.size/divisor).toLocaleString("es-PE",{maximumFractionDigits:2})} ${unit}`;
  }
  text("file-size",size);
  $("file-error").hidden=true;
  $("file").removeAttribute("aria-invalid");
}
const dropZone=$("file-picker");
let dragDepth=0;
const hasDraggedFiles=event=>Array.from(event.dataTransfer?.types || []).includes("Files");
function clearFileDrag() {
  dragDepth=0;
  dropZone.classList.remove("is-dragging");
}
dropZone.addEventListener("dragenter",event=>{
  if(!hasDraggedFiles(event) || dropZone.hidden) return;
  event.preventDefault();
  dragDepth++;
  dropZone.classList.add("is-dragging");
});
dropZone.addEventListener("dragover",event=>{
  if(!hasDraggedFiles(event) || dropZone.hidden) return;
  event.preventDefault();
  event.dataTransfer.dropEffect="copy";
});
dropZone.addEventListener("dragleave",()=>{
  dragDepth=Math.max(0,dragDepth-1);
  if(!dragDepth) clearFileDrag();
});
dropZone.addEventListener("drop",event=>{
  if(!hasDraggedFiles(event) || dropZone.hidden) return;
  event.preventDefault();
  event.stopPropagation();
  clearFileDrag();
  const files=Array.from(event.dataTransfer.files);
  const file=files[0];
  if(files.length!==1 || !file.name.toLowerCase().endsWith(".zip") || file.size>512000000) {
    text("file-error",files.length>1 ? "Arrastre un solo archivo ZIP a la vez." : "Seleccione un ZIP de hasta 512 MB.");
    $("file-error").hidden=false;
    $("file").setAttribute("aria-invalid","true");
    $("choose-file").focus();
    return;
  }
  const transfer=new DataTransfer();
  transfer.items.add(file);
  $("file").files=transfer.files;
  updateFileSelection();
  $("remove-file").focus();
});
// Prevent the browser from opening files dropped outside the green area.
document.addEventListener("dragover",event=>{
  if(!hasDraggedFiles(event)) return;
  event.preventDefault();
  if(dropZone.hidden || !dropZone.contains(event.target)) {
    event.dataTransfer.dropEffect="none";
    clearFileDrag();
  }
});
document.addEventListener("drop",event=>{
  if(hasDraggedFiles(event)) event.preventDefault();
  clearFileDrag();
});
document.addEventListener("dragend",clearFileDrag);
window.addEventListener("blur",clearFileDrag);
$("choose-file").addEventListener("click",()=>$("file").click());
$("file").addEventListener("change",()=>{
  updateFileSelection();
  if($("file").files.length) $("remove-file").focus();
});
$("remove-file").addEventListener("click",()=>{
  $("file").value="";
  updateFileSelection();
  $("choose-file").focus();
});
updateFileSelection();
$("form").addEventListener("submit",async event=>{
  event.preventDefault();
  for(const id of ["file-error","form-error"]) $(id).hidden=true;
  $("file").removeAttribute("aria-invalid");
  const file=$("file").files[0];
  if(!file || !file.name.toLowerCase().endsWith(".zip") || file.size>512000000) {
    text("file-error",!file ? "Seleccione su archivo ZIP para continuar." : "Seleccione un ZIP de hasta 512 MB.");$("file-error").hidden=false;$("file").setAttribute("aria-invalid","true");($("file").files.length ? $("remove-file") : $("choose-file")).focus();return;
  }
  $("go").disabled=true;text("go","Subiendo y revisando imágenes…");$("form").setAttribute("aria-busy","true");
  try {
    const form=new FormData($("form"));form.set("pretrained",String($("pretrained").checked));
    // Una carga grande puede tardar más que las consultas de seguimiento.
    const created=await (await request("/api/runs",{method:"POST",body:form,signal:AbortSignal.timeout(600000)})).json();
    await openRun(created.run_id); await refreshHistory();
  } catch(error) { const target=error.field==="file" ? "file-error" : "form-error";text(target,error.message);$(target).hidden=false;if(target==="file-error"){$("file").setAttribute("aria-invalid","true");($("file").files.length ? $("remove-file") : $("choose-file")).focus();} }
  finally { $("go").disabled=false;text("go","Crear mi clasificador →");$("form").removeAttribute("aria-busy"); }
});
async function action(name) {
  if(actionPending) return;
  actionPending=true;$("action-error").hidden=true;$("cancel").disabled=true;$("resume").disabled=true;$("result-resume").disabled=true;
  try { await request(`/api/runs/${runId}/${name}`,{method:"POST"}); await poll(); if(name==="resume") showView("tracking"); await refreshHistory(); }
  catch(error){text("action-error",error.message);$("action-error").hidden=false;}
  finally {actionPending=false;if(data) render(data);}
}
$("cancel").addEventListener("click",()=>action("cancel"));$("resume").addEventListener("click",()=>action("resume"));
$("result-resume").addEventListener("click",()=>action("resume"));
$("see-results").addEventListener("click",()=>showView("results"));$("open-current").addEventListener("click",()=>showView("tracking"));
$("new-project").addEventListener("click",()=>showView("start"));$("refresh-history").addEventListener("click",refreshHistory);
document.querySelectorAll("[data-view]").forEach(button=>button.addEventListener("click",()=>showView(button.dataset.view)));
$("report-details").addEventListener("toggle",async()=>{
  if(!$("report-details").open || reportRun===runId) return;
  const id=runId;text("report","Cargando reporte…");
  try {
    const [report,{renderReport}]=await Promise.all([
      request(`/api/runs/${id}/report`).then(response=>response.text()),
      import("/assets/report-markdown.js?v=20261030-markdown")
    ]);
    if(id===runId) {renderReport($("report"),report);reportRun=id;}
  }
  catch(error){if(id===runId) text("report",`${error.message} Cierre y abra esta sección para reintentar.`);}
});
function restore() {const [,next,id]=location.hash.split(/[\/#]/);const selected=id || remembered();if(selected)openRun(selected,["start","tracking","results"].includes(next)?next:"tracking",false);else showView("start",false);}
window.addEventListener("hashchange",restore);refreshHistory();restore();
