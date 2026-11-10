"""
photonic_mesh.py  --  Reck / Clements decomposition of unitaries into MZI meshes.

Self-contained (NumPy only; no `pnn` package, no git clone, no network).

Conventions
-----------
Phase shifter   P(x)         = diag(exp(i x), 1)
Balanced coupler B           = [[1, i], [i, 1]] / sqrt(2)
Mach-Zehnder    T(theta,phi) = B @ P(theta) @ B @ P(phi)
                             = i e^{i theta/2} [[ s e^{i phi},  c ],
                                                [ c e^{i phi}, -s ]]
                with s = sin(theta/2), c = cos(theta/2)

Every decomposition has the SAME output format

        U = diag(exp(i * alphas)) @ T_1 @ T_2 @ ... @ T_K

where T_k = T(theta_k, phi_k) acts on the neighbouring modes (m_k, m_k+1).
Light enters on the right (T_K first) and leaves through the output phase
screen D on the left.  Reck uses K = N(N-1)/2 MZIs in a triangle (depth 2N-3),
Clements uses the same number in a rectangle (depth N).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

_B = np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2)
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# 1.  Optical building blocks
# --------------------------------------------------------------------------- #
def coupler() -> np.ndarray:
    return _B.copy()


def P(x):
    """Phase shifter on the upper arm."""
    return np.diag([np.exp(1j * x), 1.0]).astype(complex)


def mzi(theta, phi) -> np.ndarray:
    """Ideal MZI  B P(theta) B P(phi)  (closed form, exactly unitary)."""
    s, c = np.sin(theta / 2), np.cos(theta / 2)
    g = 1j * np.exp(1j * theta / 2)
    return g * np.array([[s * np.exp(1j * phi), c],
                         [c * np.exp(1j * phi), -s]])


def random_unitary(n: int, rng=None) -> np.ndarray:
    """Haar-distributed unitary (QR of a Ginibre matrix + phase fix)."""
    rng = np.random.default_rng(rng)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    return Q * (d / np.abs(d))


# --------------------------------------------------------------------------- #
# 2.  Closed-form 2x2 solvers (replace the numerical BFGS search)
# --------------------------------------------------------------------------- #
def mzi_null(a, b, which: int = 1):
    """
    (theta, phi) such that  T(theta, phi) @ [a, b]  has output `which` equal to 0.
    The surviving output has magnitude hypot(|a|, |b|).
    """
    aa, bb = abs(a), abs(b)
    if aa < _EPS and bb < _EPS:
        return 0.0, 0.0
    if which == 1:
        return 2 * np.arctan2(aa, bb), np.angle(b) - np.angle(a)
    return 2 * np.arctan2(bb, aa), np.angle(b) - np.angle(a) + np.pi


def optical_elimination(a, b):
    """
    Two-mode block (T followed by an output phase) taking [a, b] -> [r, 0] with
    r = hypot(|a|, |b|) real and positive.  Returns (matrix, (theta, phi, gamma)).
    """
    theta, phi = mzi_null(a, b, 1)
    T = mzi(theta, phi)
    c = (T @ np.array([a, b]))[0]
    gamma = -np.angle(c) if abs(c) > _EPS else 0.0
    return P(gamma) @ T, (theta, phi, gamma)


def factor_2x2(W):
    """
    Closed-form factorisation of any 2x2 unitary  W = diag(e1, e2) @ T(theta, phi).
    Returns (e1, e2, theta, phi).  Numerically stable also for theta -> 0, pi.
    """
    theta = 2 * np.arctan2(abs(W[0, 0]), abs(W[0, 1]))
    s, c = np.sin(theta / 2), np.cos(theta / 2)
    g = 1j * np.exp(1j * theta / 2)
    if s < _EPS or c < _EPS:                      # bar / cross state: phi is free
        phi = 0.0
    else:
        phi = np.angle(W[0, 0]) - np.angle(W[0, 1])
    ephi = np.exp(1j * phi)
    if c >= s:
        e1 = W[0, 1] / (c * g)
        e2 = W[1, 0] / (c * g * ephi)
    else:
        e1 = W[0, 0] / (s * g * ephi)
        e2 = -W[1, 1] / (s * g)
    return e1, e2, theta, phi


# --------------------------------------------------------------------------- #
# 3.  Mesh container
# --------------------------------------------------------------------------- #
@dataclass
class Mesh:
    n: int
    modes: np.ndarray      # (K,) upper mode index m of each MZI (acts on m, m+1)
    thetas: np.ndarray     # (K,)
    phis: np.ndarray       # (K,)
    alphas: np.ndarray     # (N,) output phase screen
    name: str = ""

    @property
    def K(self) -> int:
        return len(self.modes)

    def matrix(self, **loss) -> np.ndarray:
        return reconstruct(self, **loss)

    def layers(self) -> np.ndarray:
        """ASAP schedule: layer index (0-based) of every MZI (light order)."""
        last = np.zeros(self.n, dtype=int)
        out = np.zeros(self.K, dtype=int)
        for k in range(self.K - 1, -1, -1):          # light enters at k = K-1
            m = self.modes[k]
            out[k] = max(last[m], last[m + 1])
            last[m] = last[m + 1] = out[k] + 1
        return out

    @property
    def depth(self) -> int:
        return int(self.layers().max()) + 1 if self.K else 0


# --------------------------------------------------------------------------- #
# 4.  Forward model (vectorised over phase-error trials, optional loss)
# --------------------------------------------------------------------------- #
def _lin(loss_dB):
    return 10.0 ** (-np.asarray(loss_dB, dtype=float) / 10.0)


def reconstruct(mesh: Mesh, thetas=None, phis=None, Lc_dB=0.0, Lp_dB=0.0,
                mzi_loss_dB=0.0):
    """
    Build the transfer matrix of the mesh.

    thetas/phis may be (K,) or (B, K); a batch gives a (B, N, N) result, which
    is how the Monte-Carlo studies avoid Python loops over trials.

    Loss model (loss values are POSITIVE dB):
      * every directional coupler transmits a power fraction 10^(-Lc_dB/10)
      * every phase shifter attenuates the arm it sits on by 10^(-Lp_dB/10)
      * mzi_loss_dB is a shortcut: total coupler loss per MZI, split evenly over its
        two couplers (== Lc_dB = mzi_loss_dB / 2).  This is the model that
        reproduces the pnn-based numbers of the original Clements notebook.
    """
    Lc_dB = np.asarray(Lc_dB, dtype=float) + np.asarray(mzi_loss_dB, dtype=float) / 2
    th = mesh.thetas if thetas is None else thetas
    ph = mesh.phis if phis is None else phis
    th, ph = np.atleast_2d(th), np.atleast_2d(ph)
    Bn = th.shape[0]
    N = mesh.n
    ac = np.sqrt(_lin(Lc_dB))                    # amplitude factor / coupler
    ap = np.sqrt(_lin(Lp_dB))                    # amplitude factor / shifter arm
    Bc = ac * _B

    M = np.broadcast_to(np.eye(N, dtype=complex), (Bn, N, N)).copy()
    for k in range(mesh.K - 1, -1, -1):
        m = mesh.modes[k]
        Pt = np.zeros((Bn, 2, 2), complex)
        Pp = np.zeros((Bn, 2, 2), complex)
        Pt[:, 0, 0], Pt[:, 1, 1] = ap * np.exp(1j * th[:, k]), 1.0
        Pp[:, 0, 0], Pp[:, 1, 1] = ap * np.exp(1j * ph[:, k]), 1.0
        T = Bc @ Pt @ Bc @ Pp                      # (B, 2, 2)
        M[:, m:m + 2, :] = T @ M[:, m:m + 2, :]
    M = np.exp(1j * mesh.alphas)[None, :, None] * M
    return M[0] if Bn == 1 else M


# --------------------------------------------------------------------------- #
# 5.  Decompositions
# --------------------------------------------------------------------------- #
def _left(M, m, col):
    """Null M[m+1, col] with an MZI on rows (m, m+1); update M in place."""
    th, ph = mzi_null(M[m, col], M[m + 1, col], which=1)
    M[m:m + 2, :] = mzi(th, ph) @ M[m:m + 2, :]
    return th, ph


def _right(M, row, col):
    """Null M[row, col] with an MZI on columns (col, col+1); update M in place."""
    # (r T^dagger)[0] = 0   <=>   T r^dagger has first component 0
    th, ph = mzi_null(np.conj(M[row, col]), np.conj(M[row, col + 1]), which=0)
    M[:, col:col + 2] = M[:, col:col + 2] @ mzi(th, ph).conj().T
    return th, ph


def _push_diagonal_left(d, left_ops):
    """Rewrite  T_1^† ... T_k^† D  as  D' T'_1 ... T'_k  (Clements' trick)."""
    d = d.copy()
    out = []
    for m, th, ph in reversed(left_ops):
        W = mzi(th, ph).conj().T @ np.diag(d[m:m + 2])
        e1, e2, th2, ph2 = factor_2x2(W)
        d[m], d[m + 1] = e1, e2
        out.append((m, th2, ph2))
    return d, out[::-1]


def _pack(n, ops, d, name):
    ops = np.array(ops, dtype=float).reshape(-1, 3)
    wrap = lambda x: np.mod(x, 2 * np.pi)
    return Mesh(n, ops[:, 0].astype(int), wrap(ops[:, 1]), wrap(ops[:, 2]),
                np.angle(d), name)


def decompose_reck(U) -> Mesh:
    U = np.asarray(U, dtype=complex)
    n = U.shape[0]
    M, left = U.copy(), []
    for col in range(n - 1):
        for r in range(n - 1, col, -1):
            th, ph = _left(M, r - 1, col)
            left.append((r - 1, th, ph))
    d, ops = _push_diagonal_left(np.diag(M), left)
    return _pack(n, ops, d, "Reck")


def decompose_clements(U) -> Mesh:
    U = np.asarray(U, dtype=complex)
    n = U.shape[0]
    M, left, right = U.copy(), [], []
    for i in range(n - 1):
        if i % 2 == 0:
            for j in range(i + 1):
                row, col = n - 1 - j, i - j
                th, ph = _right(M, row, col)
                right.append((col, th, ph))
        else:
            for j in range(i + 1):
                row, col = n - i + j - 1, j
                th, ph = _left(M, row - 1, col)
                left.append((row - 1, th, ph))
    d, ops = _push_diagonal_left(np.diag(M), left)
    # U = D' T'_left ... T'_right_last ... T_right_first
    ops = ops + right[::-1]
    return _pack(n, ops, d, "Clements")


# --------------------------------------------------------------------------- #
# 6.  Metrics
# --------------------------------------------------------------------------- #
def fidelity(U, A):
    """Normalised matrix fidelity |tr(U^† A)|^2 / (|U|_F^2 |A|_F^2); batch-aware."""
    A = np.asarray(A)
    num = np.abs(np.einsum("ij,...ij->...", U.conj(), A)) ** 2
    den = np.linalg.norm(U) ** 2 * np.sum(np.abs(A) ** 2, axis=(-2, -1))
    return num / den


def transmission(A):
    """Mean output power for unit-power single-mode inputs."""
    A = np.asarray(A)
    return np.sum(np.abs(A) ** 2, axis=(-2, -1)) / A.shape[-1]


def unitarity_error(A):
    A = np.asarray(A)
    I = np.eye(A.shape[-1])
    return np.linalg.norm(np.swapaxes(A.conj(), -1, -2) @ A - I, axis=(-2, -1))


# --------------------------------------------------------------------------- #
# 7.  Small helper used by the tutorial notebook
# --------------------------------------------------------------------------- #
def embed(T, n: int, m: int) -> np.ndarray:
    """Embed a 2x2 block T on modes (m, m+1) of an n-mode identity."""
    G = np.eye(n, dtype=complex)
    G[m:m + 2, m:m + 2] = T
    return G
