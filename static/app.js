const input = document.querySelector('#images');
const zone = document.querySelector('#drop-zone');
const list = document.querySelector('#file-list');
const button = document.querySelector('#stitch');
const status = document.querySelector('#status');
let selected = [];
function showFiles(files) { selected = [...files].slice(0, 6); list.innerHTML = selected.map((f, i) => `<div class="file"><img src="${URL.createObjectURL(f)}" alt="Preview ${i + 1}"><span>${i + 1}. ${f.name}</span></div>`).join(''); }
input.addEventListener('change', () => showFiles(input.files));
['dragenter','dragover'].forEach(event => zone.addEventListener(event, e => { e.preventDefault(); zone.classList.add('dragging'); }));
['dragleave','drop'].forEach(event => zone.addEventListener(event, e => { e.preventDefault(); zone.classList.remove('dragging'); }));
zone.addEventListener('drop', event => showFiles(event.dataTransfer.files));
for (const id of ['ratio','threshold']) document.querySelector('#' + id).addEventListener('input', e => document.querySelector('#' + id + '-value').textContent = id === 'threshold' ? e.target.value + ' px' : e.target.value);
button.addEventListener('click', async () => { if (selected.length < 2) { status.textContent = 'Select at least two images first.'; return; } const data = new FormData(); selected.forEach(file => data.append('images', file)); ['method','blend','ratio','threshold'].forEach(id => data.append(id, document.querySelector('#' + id).value)); button.disabled = true; status.textContent = 'Finding features, matching frames, and blending…'; document.querySelector('#result').classList.add('hidden'); try { const response = await fetch('/api/stitch', { method:'POST', body:data }); const result = await response.json(); if (!response.ok) throw Error(result.error || 'Stitching failed'); const image = document.querySelector('#panorama'); image.src = result.image; document.querySelector('#download').href = result.image; document.querySelector('#diagnostics').innerHTML = result.diagnostics.map(s => `<div class="metric">Step ${s.step}<b>${s.num_inliers}/${s.num_matches} inliers</b>${s.inlier_ratio * 100}% reliable · ${s.reprojection_error_px}px error</div>`).join(''); document.querySelector('#result').classList.remove('hidden'); status.textContent = `${result.width} × ${result.height} panorama created.`; } catch (error) { status.textContent = error.message; } finally { button.disabled = false; } });
