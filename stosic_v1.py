from __future__ import annotations

"""
https://github.com/gajaka/luces-pvs-theories
"""

"""
stosic_v1.py — Fisher–Voronoi na Loto 7/39 (GHQ) — 7-node krug

Izvor (Stosić / LUCES):
  luces-pvs-theories-main/fisher_voronoi.pvs
  — State, d_F, centroid(r), nearest_regime(s)

Za 7/39 (7-node):
  K=7 režima / centroida — jedna ćelija po „čvoru“ uz 7 brojeva
  State = izvlačenje kao mera na {1..39}
  d_F = 2·arccos(⟨√p, √q⟩)
  seed = 39; ceo CSV; bez randoma
  next = top 7 frekvencije u ćeliji poslednjeg izvlačenja
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np

SEED = 39
MAX_NUM = 39
N_PICK = 7
K_REGIMES = 7  # 7-node (LUCES ima 4; mi prilagođavamo na 7/39)
EPS = 1e-9

ROOT = Path(__file__).resolve().parent
GHQ = ROOT.parent
CSV_PATH = GHQ / "data" / "loto7_4652_k57.csv"


def load_draws(path: Path = CSV_PATH) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", dtype=int)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    assert raw.shape[1] == N_PICK, raw.shape
    assert raw.min() >= 1 and raw.max() <= MAX_NUM
    return raw


def draw_to_prob(draw: np.ndarray) -> np.ndarray:
    p = np.full(MAX_NUM, EPS, dtype=np.float64)
    for n in draw:
        p[int(n) - 1] = 1.0 / N_PICK
    return p / p.sum()


def fisher_rao(p: np.ndarray, q: np.ndarray) -> float:
    a = np.sqrt(np.clip(p, EPS, None))
    b = np.sqrt(np.clip(q, EPS, None))
    cos = float(np.dot(a, b))
    cos = min(1.0, max(-1.0, cos))
    return 2.0 * float(np.arccos(cos))


def nearest_regime(p: np.ndarray, centroids: np.ndarray) -> int:
    dists = [fisher_rao(p, centroids[r]) for r in range(len(centroids))]
    return int(np.argmin(dists))


def _farthest_point_init(probs: np.ndarray, k: int) -> List[int]:
    n = len(probs)
    chosen = [SEED % n]
    for _ in range(1, k):
        best_i = 0
        best_d = -1.0
        for i in range(n):
            d_min = min(fisher_rao(probs[i], probs[j]) for j in chosen)
            if d_min > best_d or (d_min == best_d and i < best_i):
                best_d = d_min
                best_i = i
        chosen.append(best_i)
    return chosen


def kmeans_fisher(
    probs: np.ndarray, k: int = K_REGIMES, max_iter: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(probs)
    chosen = _farthest_point_init(probs, k)
    centroids = probs[np.array(chosen)].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        for i in range(n):
            labels[i] = nearest_regime(probs[i], centroids)
        new_c = centroids.copy()
        for r in range(k):
            members = probs[labels == r]
            if len(members) == 0:
                others = [q for q in range(k) if q != r and np.sum(labels == q) > 0]
                if not others:
                    new_c[r] = probs[chosen[r]]
                else:
                    best_i, best_d = 0, -1.0
                    for i in range(n):
                        d_min = min(fisher_rao(probs[i], centroids[q]) for q in others)
                        if d_min > best_d or (d_min == best_d and i < best_i):
                            best_d = d_min
                            best_i = i
                    new_c[r] = probs[best_i].copy()
            else:
                m = members.mean(axis=0)
                m = np.clip(m, EPS, None)
                new_c[r] = m / m.sum()
        if np.allclose(new_c, centroids, atol=1e-10):
            centroids = new_c
            break
        centroids = new_c

    for i in range(n):
        labels[i] = nearest_regime(probs[i], centroids)
    return centroids, labels


def cell_frequency(draws: np.ndarray, labels: np.ndarray, regime: int) -> np.ndarray:
    freq = np.zeros(MAX_NUM, dtype=np.float64)
    for draw in draws[labels == regime]:
        for n in draw:
            freq[int(n) - 1] += 1.0
    return freq


def predict_next(draws: np.ndarray) -> List[int]:
    probs = np.vstack([draw_to_prob(d) for d in draws])
    _, labels = kmeans_fisher(probs, k=K_REGIMES)
    freq = cell_frequency(draws, labels, int(labels[-1]))
    order = np.lexsort((np.arange(1, MAX_NUM + 1), -freq))
    return sorted((order[:N_PICK] + 1).tolist())


def main():
    print(predict_next(load_draws()))


if __name__ == "__main__":
    main()



"""
[1, x, 7, y, 25, z, 34]
"""



"""
cd /stosic
python stosic_v1.py
"""



"""
Fisher–Voronoi (fisher_voronoi.pvs) — režimi kao ćelije, stanje = izvlačenje kao raspodela na 1..39. 

Fisher–Voronoi iz Stosića → 7 režima na celom CSV → NEXT = top 7 iz ćelije poslednjeg kola.
"""



"""
Stosić jezgro 

v1
fisher_voronoi (d_F, centroid, nearest_regime)

v2
regime_sequence / prelaz iz istog PVS
operaciona frekvencija prelaza je moja mapa, ali ideja je njegova

v3
dual_observability (obe karte)
Delimično — A+B = remix v1+v2 signala, ne čitanje PVS formula

v4
blizina u prostoru mera / stabilnost
Slabo — 1/(ε+d_F) težine su moja konstrukcija, ne direktan teorem

v5
entropy_along_geodesic (H na mid)
filter po H(mid)≤max je blizu teoremima

v6
velocity_asymmetry / lie shape direction
√p razlike; suma max(u,0) je moja agregacija
"""



"""
21 teorija

fisher_voronoi → v1, v2
dual_observability → v3
v4 se pozivao na W₂/stabilnost — slabo / nije strogo
entropy_along_geodesic → v5
velocity_asymmetry (+ delom lie_generator_structure) → v6
brenier_uniqueness (+ delom rank_orientation) → v7

kantorovich_duality
cyclical_monotonicity
displacement_interpolation
displacement_concavity
wasserstein_metric (strogo)
transport_structure
transport_structure_v2
transport_stability
stability_of_maps
monge_kantorovich_equivalence
lie_generator_structure (pun T10)
fisher_boundary
hybrid_observability
tangent_bundle
global_optimality
"""



"""
Kratko, o repou:

21 PVS teorija — sve su prošle kroz v1–v22 (neke ranije labavo: naročito v3/v4; rank_orientation je ušao uz Brenier u v7).
Repo je o spektralnom OT / LUCES (ESP32), ne o lotou — 7/39 je naša mapa, ne Stosićev domen.
Najčistije jezgro oko Fisher–Voronoi, Brenier/CM, W₂, T10 (lie_generator_structure). global_optimality je samo aksiomi + lema (bez teorema).
Empirija u PVS-u (bootovi, κ, Monge fraction) ne prenosi se automatski na CSV — samo struktura ideja.
"""
