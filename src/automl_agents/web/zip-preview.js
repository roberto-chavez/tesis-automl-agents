"use strict";
// Read only the ZIP directory and three selected images, off the UI thread.
const MAX_IMAGE = 12_000_000;
async function read(file, offset, length) {
  if(offset < 0 || length < 0 || offset + length > file.size) throw Error("Invalid ZIP range");
  return new Uint8Array(await file.slice(offset, offset + length).arrayBuffer());
}
async function entries(file) {
  const tail = await read(file, Math.max(0, file.size - 65557), Math.min(file.size, 65557));
  const view = new DataView(tail.buffer);
  let end = tail.length - 22;
  while(end >= 0 && !(view.getUint32(end, true) === 0x06054b50 && end + 22 + view.getUint16(end + 20, true) === tail.length)) end--;
  if(end < 0 || view.getUint16(end + 4, true) || view.getUint16(end + 6, true)) throw Error("Unsupported ZIP");
  const count = view.getUint16(end + 10, true);
  const size = view.getUint32(end + 12, true);
  if(count > 50000 || size > 16_000_000) throw Error("ZIP directory too large");
  const directory = await read(file, view.getUint32(end + 16, true), size);
  const dv = new DataView(directory.buffer);
  const found = [];
  for(let n = 0, at = 0; n < count; n++) {
    if(at + 46 > directory.length || dv.getUint32(at, true) !== 0x02014b50) throw Error("Invalid ZIP directory");
    const nameLength = dv.getUint16(at + 28, true);
    const next = at + 46 + nameLength + dv.getUint16(at + 30, true) + dv.getUint16(at + 32, true);
    if(next > directory.length) throw Error("Invalid ZIP entry");
    const name = new TextDecoder().decode(directory.subarray(at + 46, at + 46 + nameLength));
    const parts = name.split("/");
    if(parts.length > 1 && parts.every(p => p && !p.startsWith(".") && !p.startsWith("__")) && /\.(jpe?g|png|bmp|gif)$/i.test(name)) {
      found.push({name, parts, flags:dv.getUint16(at + 8, true), method:dv.getUint16(at + 10, true),
        compressed:dv.getUint32(at + 20, true), size:dv.getUint32(at + 24, true), offset:dv.getUint32(at + 42, true)});
    }
    at = next;
  }
  // Ignore shared wrapper folders (for example dataset/cats and dataset/dogs).
  let depth = 0;
  while(found.length && found.every(e => e.parts.length > depth + 2 && e.parts[depth] === found[0].parts[depth])) depth++;
  found.sort((a,b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
  const classes = new Map();
  for(const entry of found) {
    const label = entry.parts[depth];
    if(!classes.has(label)) classes.set(label, entry);
  }
  return [...classes].slice(0, 3);
}
async function thumbnail(file, entry) {
  if(entry.flags & 1 || entry.size > MAX_IMAGE || entry.compressed > MAX_IMAGE || ![0,8].includes(entry.method)) throw Error("Unsupported image entry");
  const header = new DataView((await read(file, entry.offset, 30)).buffer);
  if(header.getUint32(0, true) !== 0x04034b50) throw Error("Invalid image entry");
  const start = entry.offset + 30 + header.getUint16(26, true) + header.getUint16(28, true);
  const compressed = await read(file, start, entry.compressed);
  let bytes = compressed;
  if(entry.method === 8) {
    const reader = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("deflate-raw")).getReader();
    const chunks = [];
    let length = 0;
    try {
      while(true) {
        const {done, value} = await reader.read();
        if(done) break;
        length += value.length;
        if(length > MAX_IMAGE || length > entry.size) throw Error("Image too large");
        chunks.push(value);
      }
    } finally { await reader.cancel(); }
    bytes = new Blob(chunks);
  }
  const bitmap = await createImageBitmap(bytes instanceof Blob ? bytes : new Blob([bytes]));
  try {
    const canvas = new OffscreenCanvas(96, 96);
    const ctx = canvas.getContext("2d");
    const side = Math.min(bitmap.width, bitmap.height);
    ctx.drawImage(bitmap, (bitmap.width-side)/2, (bitmap.height-side)/2, side, side, 0, 0, 96, 96);
    return await canvas.convertToBlob({type:"image/webp", quality:0.8});
  } finally { bitmap.close(); }
}
self.onmessage = async ({data:file}) => {
  try {
    for(const [label, entry] of await entries(file)) {
      try { self.postMessage({label, blob:await thumbnail(file, entry)}); } catch { /* Preview failures do not block uploading. */ }
    }
  } catch { /* Invalid or unsupported archives keep the file icon. */ }
  self.postMessage({done:true});
};
