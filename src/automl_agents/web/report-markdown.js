import { marked } from "./marked.esm.js";
import DOMPurify from "./purify.es.js";

export function renderReport(container, markdown) {
  const html=marked.parse(markdown,{gfm:true,async:false});
  const fragment=DOMPurify.sanitize(html,{
    RETURN_DOM_FRAGMENT:true,
    ALLOWED_TAGS:["h1","h2","h3","h4","h5","h6","p","strong","em","del","ul","ol","li","blockquote","pre","code","hr","br","table","thead","tbody","tr","th","td","a"],
    ALLOWED_ATTR:["href","title","start","colspan","rowspan"],
    ALLOW_DATA_ATTR:false,
    ALLOW_ARIA_ATTR:false
  });
  container.replaceChildren(fragment);
  for(const table of container.querySelectorAll("table")) {
    const wrapper=document.createElement("div");
    wrapper.className="report-table";
    wrapper.tabIndex=0;
    wrapper.setAttribute("role","region");
    wrapper.setAttribute("aria-label","Tabla del reporte, desplazable horizontalmente");
    table.replaceWith(wrapper);
    wrapper.append(table);
  }
}
