import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from figdata import *
plt.rcParams.update({"font.size":7,"axes.titlesize":7.5,"axes.labelsize":7,"xtick.labelsize":6.5,"ytick.labelsize":6.5,
    "font.family":"serif","axes.edgecolor":"#52514e","axes.linewidth":0.6,"xtick.color":"#52514e","ytick.color":"#52514e",
    "text.color":"#0b0b0b","axes.labelcolor":"#0b0b0b","pdf.fonttype":42})
INK2="#52514e"; GRID="#e4e3df"; BLUE="#2a78d6"; ORANGE="#eb6834"
import os
OUT = os.path.join(os.environ.get("FIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")), "")
os.makedirs(OUT, exist_ok=True)
# planned lap time of the flown reference, from the same manifest trackfig.py reads (one source for every figure)
LAP=pd.read_csv(os.path.join(os.path.dirname(TRACKDIR),"manifest_study.csv")).set_index("seed")["lap_stretched_s"]
NAMES={"wind_const":"Constant wind","payload":"Steady downward force","mass_mult":"Mass error",
       "wind_gust":"Gust","lighthouse":"Sensing degradation","combined":"Combined"}
cmap=LinearSegmentedColormap.from_list("div",["#9e2a2a","#e34948","#f3b8b2","#f0efec","#9ec5f4","#2a78d6","#104281"])

# ---------- Fig 1: crossover map ----------
fig,axs=plt.subplots(2,3,figsize=(7.0,3.3),sharex=True,sharey=True)
for ax,c in zip(axs.flat,CONDS):
    g,ls=delta_grid(c)
    im=ax.imshow(g.values,cmap=cmap,vmin=-1,vmax=1,aspect="auto",interpolation="nearest")
    # 1px surface gaps between cells
    for k in range(1,g.shape[1]): ax.axvline(k-0.5,color="#fcfcfb",lw=0.6)
    for k in range(1,g.shape[0]): ax.axhline(k-0.5,color="#fcfcfb",lw=0.6)
    ax.axvline(8.5,color="#0b0b0b",lw=0.9,ls=(0,(3,2)))
    for i,t in enumerate(ORDER):
        a,b=ls[t]
        if a is not None:
            ax.plot(COARSE.index(a),i,marker="v",ms=4.5,color="#0b0b0b",mec="#fcfcfb",mew=0.6,ls="none")
        elif b is not None:
            ax.plot(COARSE.index(b),i,marker="v",ms=4.5,mfc="#fcfcfb",mec="#0b0b0b",mew=0.8,ls="none")
    ax.set_title(NAMES[c],loc="left",pad=3)
    ax.set_xticks(range(0,11,2)); ax.set_xticklabels([f"{l:.1f}" for l in COARSE[::2]])
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([f"{t} ({LAP[t]:.1f} s)" for t in ORDER])
    ax.tick_params(length=2,width=0.5)
    for s in ax.spines.values(): s.set_visible(False)
for ax in axs[1]: ax.set_xlabel(r"intensity $\lambda$")
for ax in axs[:,0]: ax.set_ylabel("track (planned lap)")
fig.subplots_adjust(left=0.13,right=0.88,bottom=0.13,top=0.86,wspace=0.08,hspace=0.36)
cax=fig.add_axes([0.905,0.2,0.012,0.62])
cb=fig.colorbar(im,cax=cax); cb.set_label(r"$\Delta = S_{\mathrm{RL}} - S_{\mathrm{MPC}}$  (positive: learned leads)",fontsize=6.5)
cb.set_ticks([-1,-0.5,0,0.5,1]); cb.ax.tick_params(length=2,width=0.5,labelsize=6.5); cb.outline.set_visible(False)
h=[Line2D([],[],marker="v",color="#0b0b0b",ls="none",ms=4.5,label=r"$\lambda^\star(0.8)$, inside training"),
   Line2D([],[],marker="v",mfc="#fcfcfb",mec="#0b0b0b",ls="none",ms=4.5,label=r"crosses only above the edge, $\lambda^\star(1.0)$"),
   Line2D([],[],color="#0b0b0b",ls=(0,(3,2)),lw=0.9,label="training edge (0.8)")]
fig.legend(handles=h,loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(0.5,1.0),fontsize=6.5)
fig.savefig(OUT+"fig_crossover_map.pdf"); fig.savefig(OUT+"fig_crossover_map.png",dpi=200)

# ---------- Fig 2: sensing, completion vs precision ----------
M=[];R=[]
for t in ORDER:
    m,r=family_lines("lighthouse",t,FA); M.append(m); R.append(r)
M=pd.concat(M,axis=1); R=pd.concat(R,axis=1)
p=paired("lighthouse",coarse_only=True).dropna(subset=["mpc","rl"])
med=p.groupby("lam")[["mpc","rl"]].median()
x=np.array(COARSE)
fig,(a1,a2)=plt.subplots(1,2,figsize=(3.45,1.95))
for ax in (a1,a2):
    ax.axvline(0.67,color=INK2,lw=0.7,ls=(0,(4,1.5,1,1.5)))
    ax.axvline(0.8,color=INK2,lw=0.7,ls=(0,(3,2)))
    ax.grid(axis="y",color=GRID,lw=0.5); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=2,width=0.5); ax.set_xlabel(r"intensity $\lambda$"); ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
a1.fill_between(x,R.min(1),R.max(1),color=BLUE,alpha=0.12,lw=0)
a1.fill_between(x,M.min(1),M.max(1),color=ORANGE,alpha=0.12,lw=0)
a1.plot(x,R.mean(1),color=BLUE,lw=1.4,marker="o",ms=2.6,mec="#fcfcfb",mew=0.4)
a1.plot(x,M.mean(1),color=ORANGE,lw=1.4,ls=(0,(4,1.5)),marker="s",ms=2.6,mec="#fcfcfb",mew=0.4)
a1.set_ylim(0,1.05); a1.set_title("Completion",loc="left",pad=2)
a2.plot(med.index,med.rl,color=BLUE,lw=1.4,marker="o",ms=2.6,mec="#fcfcfb",mew=0.4)
a2.plot(med.index,med.mpc,color=ORANGE,lw=1.4,ls=(0,(4,1.5)),marker="s",ms=2.6,mec="#fcfcfb",mew=0.4)
a2.set_ylim(0,0.24); a2.set_title("Paired RMSE [m]",loc="left",pad=2)
a2.text(0.03,med.rl.iloc[0]+0.012,"Learned",fontsize=6.3,ha="left")
a2.text(0.03,med.mpc.iloc[0]-0.03,"MPC",fontsize=6.3,ha="left")
a1.text(0.665,0.04,"nominal",rotation=90,fontsize=5.5,color=INK2,ha="right",va="bottom")
a1.text(0.795,0.04,"edge",rotation=90,fontsize=5.5,color=INK2,ha="right",va="bottom")
hh=[Line2D([],[],color=BLUE,lw=1.4,marker="o",ms=2.6,label="Learned (family maximum)"),Line2D([],[],color=ORANGE,lw=1.4,ls=(0,(4,1.5)),marker="s",ms=2.6,label="MPC (family maximum)")]
fig.legend(handles=hh,loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(0.55,1.0),fontsize=6.3)
fig.subplots_adjust(left=0.1,right=0.98,bottom=0.22,top=0.80,wspace=0.32)
fig.savefig(OUT+"fig_sensing_completion_rmse.pdf"); fig.savefig(OUT+"fig_sensing_completion_rmse.png",dpi=250)
print("done")
