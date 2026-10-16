"""软件渲染（QPainter + numpy，画家算法 + 平行光 + 阴影面）。不依赖 OpenGL/VTK，
所以在无显卡的虚拟机、离屏截图里都能跑。零件数量几十个以内够用。"""
import numpy as np
import trimesh
from PySide6 import QtCore, QtGui

COLORS = {"Q235": (150, 160, 175), "45": (140, 150, 165), "6061": (200, 205, 215), "6063": (205, 210, 220), "SUS304": (190, 195, 200),
          "2Cr13": (170, 175, 185), "Cu": (205, 120, 70), "Brass": (210, 170, 70), "PP": (230, 230, 220), "POM": (240, 240, 240),
          "PU": (230, 160, 60), "PC": (170, 210, 235), "MIX": (120, 120, 130), "wood": (196, 150, 98), "concrete": (160, 160, 155),
          "glass": (170, 215, 230), "rubber": (50, 50, 55), "ice": (190, 230, 250)}


class Camera:
    def __init__(self, az=-50, el=25, dist=3000, target=(0, 0, 300)):
        self.az, self.el, self.dist, self.target = az, el, dist, np.array(target, float)

    def basis(self):
        a, e = np.radians(self.az), np.radians(self.el)
        d = np.array([np.cos(e) * np.sin(a), -np.cos(e) * np.cos(a), np.sin(e)])  # 相机相对目标的方向
        fwd = -d
        right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return d, right, up, fwd

    def project(self, P, w, h, fov_scale=1.5):
        d, right, up, fwd = self.basis()
        eye = self.target + d * self.dist
        rel = P - eye
        x = rel @ right; y = rel @ up; z = rel @ fwd
        f = fov_scale * min(w, h) * 0.9
        z = np.maximum(z, 1.0)
        return np.stack([w / 2 + f * x / z, h / 2 - f * y / z], axis=1), z


class SceneRenderer:
    def __init__(self, scene):
        self.set_scene(scene)

    def set_scene(self, scene):
        self.scene = scene
        self.local = []
        for p in scene.parts:
            m = p.local_mesh()
            dmax = max(p.dims)
            edge = dmax / 7 if p.kind != "sphere" else dmax
            if p.kind in ("box", "cyl", "tube"):
                v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=edge)
                m = trimesh.Trimesh(v, f, process=False)
            self.local.append((np.asarray(m.vertices), np.asarray(m.faces), np.asarray(m.face_normals)))
        self.anchor = {}
        grp = {}
        for i, p in enumerate(scene.parts):
            key = "__world__" if p.fixed else (p.group or f"__{i}")
            grp.setdefault(key, []).append(i)
        self.groups = []
        for key, idx in grp.items():
            if key == "__world__": self.groups.append((None, idx, None))
            else: self.groups.append((key, idx, np.array(scene.parts[idx[0]].pos)))
        self.body_index = {}
        bi = 0
        for key, idx, anc in self.groups:
            if key is not None:
                for i in idx: self.body_index[i] = (bi, anc)
                bi += 1

    def world_geoms(self, frame=None):
        """每个零件的 (世界顶点, 面, 面法线)。frame: (n_body, 7) [x y z qw qx qy qz] 单位 m"""
        from scipy.spatial.transform import Rotation as R
        out = []
        for i, p in enumerate(self.scene.parts):
            V, F, N = self.local[i]
            Rp = p.R
            if frame is None or i not in self.body_index:
                Rw = Rp; pos = np.array(p.pos)
            else:
                bi, anc = self.body_index[i]
                f = frame[bi]
                Rb = R.from_quat([f[4], f[5], f[6], f[3]])
                pos = f[:3] * 1000 + Rb.apply(np.array(p.pos) - anc)
                Rw = Rb * Rp
            out.append((Rw.apply(V) + pos, F, Rw.apply(N)))
        return out

    def paint(self, painter, w, h, cam, frame=None, mark=None, bg=(246, 247, 250)):
        mark = mark or {}
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(0, 0, w, h, QtGui.QColor(*bg))
        # 地面格子
        step = 500
        ext = 3000
        pen = QtGui.QPen(QtGui.QColor(205, 208, 215), 1)
        painter.setPen(pen)
        for g in np.arange(-ext, ext + 1, step):
            a, _ = cam.project(np.array([[g, -ext, 0], [g, ext, 0], [-ext, g, 0], [ext, g, 0]], float), w, h)
            painter.drawLine(QtCore.QPointF(*a[0]), QtCore.QPointF(*a[1]))
            painter.drawLine(QtCore.QPointF(*a[2]), QtCore.QPointF(*a[3]))
        # 坐标轴
        ax, _ = cam.project(np.array([[0, 0, 0], [400, 0, 0], [0, 400, 0], [0, 0, 400]], float), w, h)
        for k, col in ((1, "#d33"), (2, "#2a2"), (3, "#33d")):
            painter.setPen(QtGui.QPen(QtGui.QColor(col), 2))
            painter.drawLine(QtCore.QPointF(*ax[0]), QtCore.QPointF(*ax[k]))
        geoms = self.world_geoms(frame)
        light = np.array([0.4, -0.5, 0.8]); light /= np.linalg.norm(light)
        d_, _, _, fwd = cam.basis()
        tris = []
        for i, (V, F, N) in enumerate(geoms):
            P2, z = cam.project(V, w, h)
            facing = (N @ d_) > -0.02   # 朝向相机的面
            zf = z[F].mean(axis=1)
            base = np.array(mark.get(self.scene.parts[i].name) or COLORS.get(self.scene.parts[i].mat, (150, 150, 150)), float)
            shade = 0.45 + 0.55 * np.clip(N @ light, 0, 1)
            for k in np.nonzero(facing)[0]:
                tris.append((zf[k], P2[F[k]], base * shade[k]))
        tris.sort(key=lambda t: -t[0])
        for _, pts, col in tris:
            c = np.clip(col, 0, 255).astype(int)
            qc = QtGui.QColor(int(c[0]), int(c[1]), int(c[2]))
            painter.setPen(QtGui.QPen(qc, 0.8))
            painter.setBrush(qc)
            painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*p) for p in pts]))
        if self.scene.fluid:
            lv = float(self.scene.fluid[1]); e = 1100
            q, _ = cam.project(np.array([[-e, -e, lv], [e, -e, lv], [e, e, lv], [-e, e, lv]], float), w, h)
            painter.setPen(QtGui.QPen(QtGui.QColor(40, 120, 200, 160), 1.5)); painter.setBrush(QtGui.QColor(60, 140, 220, 70))
            painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*x) for x in q]))
        # 标签
        painter.setPen(QtGui.QColor(30, 30, 40))
        font = painter.font(); font.setPointSize(9); painter.setFont(font)
        for i, p in enumerate(self.scene.parts):
            if p.name in mark:
                V = geoms[i][0]; c = V.mean(axis=0)
                q, _ = cam.project(c[None, :], w, h)
                painter.drawText(QtCore.QPointF(q[0][0] + 4, q[0][1] - 4), p.name)

    def auto_camera(self):
        pts = np.vstack([np.array(p.mesh().vertices) for p in self.scene.parts]) if self.scene.parts else np.zeros((1, 3))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        ctr = (lo + hi) / 2
        size = np.linalg.norm(hi - lo)
        return Camera(az=-50, el=24, dist=max(1200, size * 1.35), target=(ctr[0], ctr[1], max(ctr[2] * 0.7, 100)))


def render_png(scene, path, w=900, h=600, cam=None, frame=None, mark=None):
    img = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
    p = QtGui.QPainter(img)
    r = SceneRenderer(scene)
    r.paint(p, w, h, cam or r.auto_camera(), frame, mark)
    p.end()
    img.save(path)
    return path
