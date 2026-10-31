"""Seed-level bootstrap of lambda* and crossover counts on the stochastic channels (primary line)."""
from lib import *
rng = np.random.default_rng(1)
B = 2000

def arrays(cond, tracks):
    d = load(); d = d[(d.cond == cond) & d.track.isin(tracks) & d.lam.isin(COARSE) & d.member.isin(MPC + FA)]
    out = {}
    for t in tracks:
        vm = viable_members(cond, t, MPC); vr = viable_members(cond, t, FA)
        x = d[d.track == t]
        seeds = sorted(x[x.lam > 0].seed.unique())
        A = {}
        for m in vm + vr:
            y = x[x.member == m]
            mat = np.full((len(COARSE), len(seeds)), np.nan)
            for i, l in enumerate(COARSE):
                z = y[y.lam == l].set_index("seed").completed
                if l == 0.0:
                    mat[i, :] = z.mean()            # single nominal lap: fixed
                else:
                    mat[i, :] = [z.get(s, np.nan) for s in seeds]
            A[m] = mat
        out[t] = (vm, vr, A, len(seeds))
    return out

def run(cond, tracks):
    arr = arrays(cond, tracks)
    res = {t: [] for t in tracks}; counts = []
    for b in range(B):
        n = 0
        for t in tracks:
            vm, vr, A, ns = arr[t]
            idx = rng.integers(0, ns, ns)
            mpc = np.nanmax([np.nanmean(A[m][:, idx], 1) for m in vm], 0)
            rl = np.nanmax([np.nanmean(A[m][:, idx], 1) for m in vr], 0)
            ls = lam_star(pd.Series(rl - mpc, index=COARSE), 0.8, 0.2)
            res[t].append(ls); n += ls is not None
        counts.append(n)
    return res, np.array(counts)

if __name__ == "__main__":
    P = entering_tracks("lighthouse", FA)
    for c in ["lighthouse", "combined", "wind_gust"]:
        res, counts = run(c, P)
        print(c)
        for t in P:
            v = [x for x in res[t] if x is not None]
            p = len(v) / B
            q = np.percentile(v, [2.5, 97.5]) if v else [np.nan, np.nan]
            print(f"  {t}: P(cross)={p:.2f}  lambda* 95% [{q[0]:.1f}, {q[1]:.1f}]  median {np.median(v) if v else np.nan}")
        print("  count of crossing tracks: 95%", np.percentile(counts, [2.5, 97.5]), "mean", counts.mean().round(2),
              "dist", {k: int((counts == k).sum()) for k in range(7)})
