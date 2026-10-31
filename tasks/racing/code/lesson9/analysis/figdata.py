from lib import *
from pairs import paired
ORDER=[1089,1145,1250,965,757,989]
def delta_grid(cond):
    rows={}; ls={}
    for t in ORDER:
        m,r=family_lines(cond,t,FA)
        d=r-m; rows[t]=d
        a=lam_star(d,0.8,0.2); b=lam_star(d,1.0,0.2)
        ls[t]=(a,b)
    return pd.DataFrame(rows).T, ls
if __name__=="__main__":
    for c in CONDS:
        g,ls=delta_grid(c)
        print(c, {t:ls[t] for t in ORDER})
    # pooled sensing family lines
    M=[];R=[]
    for t in ORDER:
        m,r=family_lines("lighthouse",t,FA); M.append(m); R.append(r)
    M=pd.concat(M,axis=1); R=pd.concat(R,axis=1)
    print("MPC mean", M.mean(1).round(2).to_dict()); print("RL mean", R.mean(1).round(2).to_dict())
    p=paired("lighthouse",coarse_only=True).dropna(subset=["mpc","rl"])
    print(p.groupby("lam")[["mpc","rl"]].median().round(3).T.to_string())
    print("pairs per lam", p.groupby("lam").size().to_dict())
