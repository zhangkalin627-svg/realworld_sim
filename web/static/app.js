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
  select(0);
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

function poll(jid) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    const j = await (await fetch(`/api/job/${jid}`)).json();
    $('#barfill').style.width = `${(j.progress * 100).toFixed(0)}%`;
    if (j.status === 'running') return poll(jid);
    $('#runbtn').disabled = false;
    $('#bar').style.display = 'none';
    if (j.status === 'error') {
      $('#status').innerHTML = `<span class="err">出错了：\n${j.error}</span>`;
      return;
    }
    const r = j.result;
    $('#status').textContent = '完成';
    $('#lines').textContent = r.lines.join('\n');
    $('#warns').innerHTML = (r.warnings || []).map(w => `<div>⚠ ${w}</div>`).join('');
    $('#plots').innerHTML = (r.plots || []).map(p => p.svg).join('');
  }, 700);
}

boot();
