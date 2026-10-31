from lib import *

_vcache = {}
def viab_mask(cond):
    """Boolean Series index (track, member): viable at lam0 on this channel."""
    if cond not in _vcache:
        ct = comp_table(cond)
        z = ct.xs(0.0, level="lam")
        _vcache[cond] = (z >= 0.5)
    return _vcache[cond]

def paired(cond, rl_mem=FA, tracks=None, viable=True, coarse_only=False, col="rmse_3d"):
    d = load(); d = d[d.cond == cond]
    tracks = tracks if tracks is not None else entering_tracks(cond, rl_mem)
    d = d[d.track.isin(tracks) & d.member.isin(MPC + rl_mem)].copy()
    if coarse_only:
        d = d[d.lam.isin(COARSE)]
    if viable:
        vm = viab_mask(cond)
        keys = list(zip(d.track, d.member))
        d = d[[bool(vm.get(k, False)) for k in keys]]
    d = d[d.completed == 1]
    d["fam"] = np.where(d.member.isin(MPC), "mpc", "rl")
    g = d.groupby(["track", "lam", "seed", "fam"])[col].median().unstack("fam")
    for f in ("mpc", "rl"):
        if f not in g:
            g[f] = np.nan
    return g.reset_index()

if __name__ == "__main__":
    for viable in [True, False]:
        for co in [False, True]:
            print(f"viable={viable} coarse_only={co}")
            for c in CONDS:
                p = paired(c, viable=viable, coarse_only=co).dropna(subset=["mpc", "rl"])
                print(f"  {c:11s} pairs={len(p):5d} mpc={p.mpc.median():.3f} rl={p.rl.median():.3f}")
