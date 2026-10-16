"""符号组装（sparsity pattern 一次构建、每步只更新数值）——v0.28 加速第二档。

原理（缪斯原型验证，realworld_sim_symasm_proto.py）：固定网格/基函数下 skfem asm 的
COO (rows,cols) 顺序确定；k/rc/h 恒正 ⇒ 零元位置与数值无关 ⇒ CSR pattern 可复用。
变体 S 进一步利用 P1 单元特性（∇φ 元常数）把通用核换成预计算几何张量 + einsum 缩并。

集成安全网：首次构建时在随机非均匀场上与 skfem asm 逐核对比（rel < 1e-12），
任何不一致（如 skfem 大版本改动 _assemble 语义）自动回退原通用 asm 路径。
"""
import numpy as np
from scipy.sparse import csr_matrix
from skfem import asm


class SymAssembler:
    """一个 skfem 双线性型的符号模板：pattern 一次构建，之后只更新数值。"""

    def __init__(self, form, basis, params0):
        self.form, self.basis = form, basis
        ref = asm(form, basis, **params0)
        idx, data0, shape, _ = form._assemble(basis, **params0)
        rows, cols = idx[0].astype(np.int64), idx[1].astype(np.int64)
        self.shape = shape
        nz = data0 != 0
        r, c = rows[nz], cols[nz]
        order = np.lexsort((c, r))
        r_s, c_s = r[order], c[order]
        new_seg = np.empty(len(r_s), bool); new_seg[0] = True
        new_seg[1:] = (r_s[1:] != r_s[:-1]) | (c_s[1:] != c_s[:-1])
        seg = np.cumsum(new_seg) - 1
        nnz = int(seg[-1]) + 1
        assert nnz == ref.nnz, f"nnz mismatch {nnz} vs {ref.nnz}"
        starts = np.where(new_seg)[0]
        row_of = r_s[starts]
        indptr = np.zeros(ref.shape[0] + 1, dtype=np.int64)
        np.add.at(indptr, row_of + 1, 1); np.cumsum(indptr, out=indptr)
        assert np.array_equal(indptr, ref.indptr), "indptr mismatch"
        assert np.array_equal(c_s[starts], ref.indices), "indices mismatch"
        self.coo2csr = np.empty(len(r), dtype=np.int64)
        self.coo2csr[order] = seg
        self.nz_pos = np.where(nz)[0]
        d_check = np.bincount(self.coo2csr, weights=data0[self.nz_pos], minlength=nnz)
        rd = np.abs(d_check - ref.data) / max(np.abs(ref.data).max(), 1e-300)
        assert rd.max() < 1e-15, f"template data mismatch rel={rd.max()}"
        self.indptr, self.indices, self.nnz = indptr, c_s[starts], nnz

    def scatter(self, data_coo):
        """COO 数值（与 _assemble 同序）→ CSR data。"""
        return np.bincount(self.coo2csr, weights=data_coo[self.nz_pos], minlength=self.nnz)


class SymThermalS:
    """P1 单元特化的热步核：预计算几何张量，每步只做逐点物性 + 小张量缩并。"""

    def __init__(self, sim):
        tb = sim.tb
        self.dofs = tb.element_dofs.T.copy()          # (ne,3)
        self.N = tb.N
        phi_ref = np.stack([tb.basis[j][0].value[0] for j in range(3)], axis=0)  # (3,nq)
        self.phi_ref = phi_ref
        grad = np.stack([tb.basis[j][0].grad for j in range(3)], axis=0)         # (3,2,ne,nq)
        g0 = grad.transpose(2, 0, 1, 3)[..., 0]       # (ne,3,2) P1: ∇φ 元内常数
        self.DOT = np.einsum('eia,eja->eij', g0, g0)
        r_nodes = tb.mesh.p[0]
        r_q = r_nodes[self.dofs] @ phi_ref.T          # (ne,nq)
        self.RW = r_q * tb.dx
        self.P = (phi_ref[:, None, :] * phi_ref[None, :, :])[None] * self.RW[:, None, None, :]
        self.L = np.einsum('iq,eq->ei', phi_ref, self.RW)
        self.qmap = self.dofs.reshape(-1)
        self.fb = {}
        for key, fbasis in (("o", sim.fb_outer), ("i", sim.fb_inner)):
            fphi = np.stack([fbasis.basis[j][0].value for j in range(3)], axis=1)  # (nf,3,nqf)
            fdofs = fbasis.element_dofs.T.copy()
            fr_q = np.einsum('fjq,fj->fq', fphi, r_nodes[fdofs])
            fRW = fr_q * fbasis.dx
            self.fb[key] = dict(
                fphi=fphi,
                fdofs=fdofs,
                fP=fphi[:, :, None, :] * fphi[:, None, :, :] * fRW[:, None, None, :],
                fF=fphi * fRW[:, None, :])

    def interp_vol(self, T):
        return T[self.dofs] @ self.phi_ref.T

    def kernel_K(self, k, tmpl):
        c = (k * self.RW).sum(axis=1)
        d = (self.DOT.transpose(1, 2, 0) * c).reshape(-1)
        return csr_matrix((tmpl.scatter(d), tmpl.indices, tmpl.indptr), shape=tmpl.shape)

    def kernel_C(self, rc, tmpl):
        Ce = np.einsum('eq,eijq->eij', rc, self.P)
        d = Ce.transpose(1, 2, 0).reshape(-1)
        return csr_matrix((tmpl.scatter(d), tmpl.indices, tmpl.indptr), shape=tmpl.shape)

    def kernel_Q(self, q_e):
        Qe = q_e[:, None] * self.L
        return np.bincount(self.qmap, weights=Qe.reshape(-1), minlength=self.N)

    def interp_fb(self, key, T):
        f = self.fb[key]
        return np.einsum('fjq,fj->fq', f["fphi"], T[f["fdofs"]])

    def kernel_H(self, key, h, tmpl):
        He = np.einsum('fq,fijq->fij', h, self.fb[key]["fP"])
        d = He.transpose(1, 2, 0).reshape(-1)
        return csr_matrix((tmpl.scatter(d), tmpl.indices, tmpl.indptr), shape=tmpl.shape)

    def kernel_hlin(self, key, h, Tinf):
        f = self.fb[key]
        be = Tinf * np.einsum('fq,fjq->fj', h, f["fF"])
        return np.bincount(f["fdofs"].reshape(-1), weights=be.reshape(-1), minlength=self.N)
