const $ = s => document.querySelector(s);
let MODULES = [], cur = null, pollTimer = null;

async function boot() {
  MODULES = await (await fetch('/api/modules')).json();
  const nav = $('#modtabs');
  MODULES.forEach((m, i) => {
    const b = document.createElement('button');
    b.textContent = m.name;
    b.onclick = () => select(i);
    nav.appendChild(b);
  });
  window._mods = MODULES;
  const meta = await (await fetch('/api/meta')).json();
  const v = document.querySelector('#ver');
  if (v) v.textContent = `v${meta.version} · ${meta.modules} 个模块`;
  select(0);
  loadJobs();
}

function select(i) {
  cur = MODULES[i];
  document.querySelectorAll('nav button').forEach((b, j) => b.classList.toggle('on', i === j));
  $('#mtitle').textContent = cur.name;
  $('#mdesc').textContent = cur.desc || '';
  const f = $('#pform'); f.innerHTML = '';
  let fs = null;
  cur.fields.forEach(fd => {
    if (fd.g !== undefined) {
      fs = document.createElement('fieldset');
      fs.innerHTML = `<legend>${fd.g}</legend>`;
      f.appendChild(fs); return;
    }
    const host = fs || f;
    const lab = document.createElement('label');
    if (fd.type === 'check') {
      lab.className = 'ck';
      lab.innerHTML = `<input type="checkbox" name="${fd.name}" ${fd.default ? 'checked' : ''}> <span>${fd.label}</span>`;
    } else if (fd.type === 'textarea') {
      lab.className = 'ta';
      const ta = document.createElement('textarea');
      ta.name = fd.name; ta.rows = 16; ta.spellcheck = false;
      ta.value = fd.default || '';
      lab.innerHTML = `<span>${fd.label}</span>`;
      lab.appendChild(ta);
    } else if (fd.type === 'select') {
      const opts = fd.options.map(o => {
        const v = typeof o === 'object' ? o.v : o, t = typeof o === 'object' ? o.t : o;
        return `<option value="${v}" ${v === fd.default ? 'selected' : ''}>${t}</option>`;
      }).join('');
      lab.innerHTML = `<span>${fd.label}</span><select name="${fd.name}">${opts}</select>`;
    } else {
      lab.innerHTML = `<span>${fd.label}</span><input type="number" name="${fd.name}" value="${fd.default}" min="${fd.min}" max="${fd.max}" step="${fd.step || 'any'}">`;
    }
    host.appendChild(lab);
  });
}

$('#runbtn').onclick = async () => {
  const fd = new FormData($('#pform'));
  const params = {};
  cur.fields.forEach(x => {
    if (x.g !== undefined) return;
    if (x.type === 'check') params[x.name] = fd.get(x.name) === 'on';
    else if (x.type === 'number') params[x.name] = parseFloat(fd.get(x.name));
    else params[x.name] = fd.get(x.name);
  });
  $('#runbtn').disabled = true;
  $('#status').textContent = '仿真进行中 …';
  $('#lines').innerHTML = ''; $('#warns').innerHTML = ''; $('#plots').innerHTML = '';
  $('#bar').style.display = 'block';
  const r = await (await fetch(`/api/sim/${cur.key}`, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({params})
  })).json();
  poll(r.job);
};

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function renderResult(j) {
  $('#runbtn').disabled = false;
  $('#bar').style.display = 'none';
  if (j.status === 'error' || j.status === 'interrupted') {
    $('#status').innerHTML = `<span class="err">${j.status === 'interrupted' ? '任务被服务器重启中断（参数已保留，可重新提交）' : '出错了：'}\n${esc(j.error || '')}</span>`;
    if (j.has_checkpoint) {
      const b = document.createElement('button');
      b.textContent = '⏩ 从断点续算';
      b.style.marginTop = '8px';
      b.onclick = async () => {
        b.disabled = true; b.textContent = '续算提交中 …';
        const r = await (await fetch(`/api/sim/${j.module}`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({params: {...j.params, resume_from: j.id}})
        })).json();
        $('#status').textContent = '仿真进行中（断点续算）…';
        $('#bar').style.display = 'block';
        poll(r.job);
      };
      $('#status').appendChild(document.createElement('br'));
      $('#status').appendChild(b);
    }
    return;
  }
  const r = j.result;
  $('#status').textContent = '完成';
  $('#lines').textContent = r.lines.join('\n');
  $('#warns').innerHTML = (r.warnings || []).map(w => `<div>⚠ ${esc(w)}</div>`).join('');
  $('#plots').innerHTML = (r.plots || []).map(p => p.svg).join('');
  if (r.viewer) {
    const vd = document.createElement('div'); vd.id = 'v3d';
    $('#plots').appendChild(vd);
    import('/viewer3d.js').then(m => m.show(vd, r.viewer)).catch(e => {
      vd.innerHTML = '<span class="err">3D 查看器加载失败: ' + esc(e) + '</span>';
    });
  }
}

function poll(jid) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    const j = await (await fetch(`/api/job/${jid}`)).json();
    $('#barfill').style.width = `${(j.progress * 100).toFixed(0)}%`;
    if (j.status === 'running') return poll(jid);
    renderResult(j); loadJobs();
  }, 700);
}

async function loadJobs() {
  const r = await (await fetch('/api/jobs')).json();
  const host = $('#joblist');
  if (!host) return;
  if (!r.jobs.length) { host.innerHTML = '<div class="muted">还没有任务</div>'; return; }
  const names = {}; (window._mods || []).forEach(m => names[m.key] = m.name);
  host.innerHTML = r.jobs.map(j => {
    const t = new Date(j.created * 1000);
    const icon = {done: '✅', running: '⏳', error: '❌', interrupted: '⛔'}[j.status] || '?';
    return `<div class="jobrow" data-jid="${j.id}">${icon} ${esc(names[j.module] || j.module)} <span class="muted">${t.getMonth()+1}/${t.getDate()} ${String(t.getHours()).padStart(2,'0')}:${String(t.getMinutes()).padStart(2,'0')}</span></div>`;
  }).join('');
  host.querySelectorAll('.jobrow').forEach(el => el.onclick = async () => {
    const j = await (await fetch(`/api/job/${el.dataset.jid}`)).json();
    if (j.status === 'running') { $('#status').textContent = '该任务还在运行'; return; }
    $('#lines').innerHTML = ''; $('#warns').innerHTML = ''; $('#plots').innerHTML = '';
    renderResult(j);
  });
}

boot();
