// 三维装配世界查看器 (v0.18)：three.js 渲染零件 mesh，逐帧位姿动画，按体检状态着色。
import * as THREE from '/vendor/three.module.min.js';
import { OrbitControls } from '/vendor/OrbitControls.js';

const STATUS_COLOR = { '正常': null, '稳定': null, '铰接': null, '屈服': 0xe6a23c, '有位移/下落': 0xe6a23c,
                       '断裂': 0xd23c3c, '倾倒/滑落': 0xd23c3c, '固定': 0x8a8f98 };

export function show(container, data) {
  container.innerHTML = '';
  const W = container.clientWidth || 700, H = 460;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf5f6f8);
  const cam = new THREE.PerspectiveCamera(45, W / H, 1, 200000);
  const rend = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  rend.setSize(W, H); rend.setPixelRatio(window.devicePixelRatio || 1);
  container.appendChild(rend.domElement);
  const ctl = new OrbitControls(cam, rend.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const dl = new THREE.DirectionalLight(0xffffff, 1.2); dl.position.set(300, 500, 400); scene.add(dl);
  const grid = new THREE.GridHelper(2000, 20, 0x999999, 0xcccccc); scene.add(grid);   // 地面 z=0

  // 零件 mesh（物理 z 向上 → three.js y 向上：几何、位姿、四元数统一做 (x,y,z)→(x,z,-y) 换基 = 绕 x 转 -90°）
  const parts = {};
  let cx = 0, cy = 0, n = 0;
  data.parts.forEach(p => {
    const m = data.meshes[p.name];
    if (!m) return;
    const v2 = new Float32Array(m.v.length);
    for (let i = 0; i < m.v.length; i += 3) { v2[i] = m.v[i]; v2[i+1] = m.v[i+2]; v2[i+2] = -m.v[i+1]; }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(v2, 3));
    g.setIndex(m.f); g.computeVertexNormals();
    const glassy = p.glass;
    const mat = new THREE.MeshStandardMaterial({ color: p.color, roughness: glassy ? 0.15 : 0.75,
      metalness: glassy ? 0.0 : 0.25, transparent: glassy, opacity: glassy ? 0.45 : 1.0 });
    const mesh = new THREE.Mesh(g, mat);
    mesh.rotation.order = 'XYZ';
    scene.add(mesh);
    parts[p.name] = mesh;
    const c = data.poses[p.name][0];
    cx += c[0]; cy += c[1]; n++;
  });
  if (n) { cx /= n; cy /= n; }
  // 视野：以零件初始位置包围
  let rmax = 100;
  data.parts.forEach(p => { const c = data.poses[p.name][0]; rmax = Math.max(rmax, Math.hypot(c[0]-cx, c[1]-cy) * 2.2); });
  cam.position.set(cx + rmax * 1.6, rmax * 1.4, -cy + rmax * 1.6);
  ctl.target.set(cx, 0, -cy); ctl.update();

  // 状态着色（屈服橙/断裂红，正常用材质本色）
  data.parts.forEach(p => {
    const st = data.status[p.name] || (p.fixed ? '固定' : '正常');
    const c = STATUS_COLOR[st];
    if (c != null && parts[p.name]) parts[p.name].material.color.setHex(c);
  });

  // 控制条
  const bar = document.createElement('div'); bar.className = 'v3dbar';
  const btn = document.createElement('button'); btn.textContent = '▶ 播放';
  const sld = document.createElement('input'); sld.type = 'range'; sld.min = 0; sld.max = data.times.length - 1; sld.value = data.times.length - 1;
  const lab = document.createElement('span');
  bar.appendChild(btn); bar.appendChild(sld); bar.appendChild(lab);
  container.appendChild(bar);
  const legend = document.createElement('div'); legend.className = 'muted';
  legend.textContent = '拖拽旋转 / 滚轮缩放 / 右键平移；颜色：本色=正常，橙=屈服或位移，红=断裂/倾倒，玻璃半透明';
  container.appendChild(legend);

  const zUp2three = (mesh, fr) => {
    mesh.position.set(fr[0], fr[2], -fr[1]);          // (x, z, -y)
    mesh.quaternion.set(fr[3], fr[5], -fr[4], fr[6]); // 四元数同基变换 (x,y,z,w)→(x,z,-y,w)
  };
  let playing = false, raf = null, t0 = 0;
  function setFrame(i) {
    i = Math.max(0, Math.min(data.times.length - 1, i | 0));
    sld.value = i;
    data.parts.forEach(p => { if (parts[p.name]) zUp2three(parts[p.name], data.poses[p.name][i]); });
    lab.textContent = `t = ${data.times[i].toFixed(2)} s (${i + 1}/${data.times.length})`;
    rend.render(scene, cam);
  }
  function tick(ts) {
    if (!playing) return;
    if (!t0) t0 = ts;
    const f = ((ts - t0) / 1000) * 15;
    if (f >= data.times.length - 1) { playing = false; btn.textContent = '▶ 播放'; setFrame(data.times.length - 1); return; }
    setFrame(f);
    raf = requestAnimationFrame(tick);
  }
  btn.onclick = () => {
    playing = !playing; btn.textContent = playing ? '⏸ 暂停' : '▶ 播放';
    if (playing) { t0 = 0; if (+sld.value >= data.times.length - 1) setFrame(0); raf = requestAnimationFrame(tick); }
    else cancelAnimationFrame(raf);
  };
  sld.oninput = () => { playing = false; btn.textContent = '▶ 播放'; cancelAnimationFrame(raf); setFrame(+sld.value); };
  function loop() { ctl.update(); rend.render(scene, cam); if (!playing) raf = requestAnimationFrame(loop); }
  setFrame(data.times.length - 1);
  loop();
}
