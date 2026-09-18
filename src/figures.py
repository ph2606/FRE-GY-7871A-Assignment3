"""Publication figures generated only from the frozen observed inputs/results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
NAVY='#18354a';TEAL='#167d8d';RED='#c46046';GOLD='#bb923c';GRAY='#737f88'


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titleweight':'bold',
      'axes.labelcolor':NAVY,'text.color':NAVY,'axes.edgecolor':'#c9d0d5','axes.spines.top':False,
      'axes.spines.right':False,'grid.color':'#e3e7ea','grid.alpha':.8,'axes.grid':True,'axes.axisbelow':True,
      'figure.facecolor':'white','savefig.facecolor':'white','legend.frameon':False})


def dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_xlim(pd.Timestamp('2026-01-01'),pd.Timestamp('2026-09-17'))


def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight')
    plt.close(fig)


def build():
    style();OUT.mkdir(parents=True,exist_ok=True)
    levels=pd.read_csv(ROOT/'data/processed/market_levels.csv',index_col=0,parse_dates=True).loc['2026-01-01':]
    news=pd.read_csv(ROOT/'data/processed/news_daily.csv',index_col=0,parse_dates=True)
    gpr=pd.read_csv(ROOT/'data/processed/gpr_benchmark.csv',index_col=0,parse_dates=True)
    fig,axs=plt.subplots(3,1,figsize=(9,8),sharex=True,gridspec_kw={'height_ratios':[1.25,1,.9]})
    for c,label,color in [('DCOILWTICO','WTI spot',TEAL),('DCOILBRENTEU','Brent spot',NAVY)]:
        s=levels[c].dropna();axs[0].plot(s.index,s,label=label,color=color,lw=1.8)
    axs[0].set_title('Physical crude prices and the Brent–WTI spread',loc='left');axs[0].set_ylabel('USD / barrel');axs[0].legend(ncol=2,loc='upper left')
    s=levels.brent_wti_spread.dropna();axs[1].plot(s.index,s,color=RED,lw=1.7)
    axs[1].axhline(0,color=GRAY,lw=.8);axs[1].set_ylabel('Brent − WTI\nUSD / barrel')
    for c,label,color in [('CL=F_close','WTI rolling futures',TEAL),('BZ=F_close','Brent rolling futures',NAVY)]:
        s=levels[c].dropna();axs[2].plot(s.index,s,label=label,color=color,lw=1.4)
    axs[2].set_ylabel('USD / barrel');axs[2].legend(ncol=2,loc='upper left');axs[2].set_xlabel('2026 • Separate rolling futures; delivery months can differ')
    for ax in axs:dates(ax);ax.axvline(pd.Timestamp('2026-03-02'),color=GRAY,ls=':',lw=1)
    fig.tight_layout();save(fig,'figure1_oil')
    fig,axs=plt.subplots(2,1,figsize=(9,6.4),sharex=True)
    for c,label,color in [('^GSPC_close','S&P 500',NAVY),('EFA','Developed ex-US (EFA)',TEAL),('EEM','Emerging markets (EEM)',GOLD),('GLD','Gold proxy (GLD)',RED)]:
        s=levels[c].dropna();axs[0].plot(s.index,100*s/s.iloc[0],label=label,color=color,lw=1.5)
    axs[0].set_title('Global risk assets and Treasury yields',loc='left');axs[0].set_ylabel('First 2026 observation = 100');axs[0].legend(ncol=2,fontsize=8.5)
    for c,label,color in [('DGS3MO','3-month',GOLD),('DGS2','2-year',TEAL),('DGS10','10-year',NAVY)]:
        s=levels[c].dropna();axs[1].plot(s.index,s,label=label,color=color,lw=1.6)
    axs[1].set_ylabel('Yield, %');axs[1].legend(ncol=3)
    for ax in axs:dates(ax)
    fig.tight_layout();save(fig,'figure2_markets')
    fig,axs=plt.subplots(3,1,figsize=(9,8),sharex=True)
    axs[0].bar(news.index,news.war_attention,width=1.6,color=TEAL,alpha=.5,label='War articles per calendar day in session interval')
    axs[0].plot(news.index,news.war_attention.rolling(5,min_periods=3).mean(),color=NAVY,lw=1.8,label='Trailing 5-session mean')
    axs[0].set_title('Attention, physical-conflict vocabulary and a global benchmark',loc='left');axs[0].set_ylabel('Articles / day');axs[0].legend(fontsize=8)
    axs[1].plot(news.index,news.negative_pct.rolling(5,min_periods=3).mean(),color=RED,label='Negative-word share')
    axs[1].plot(news.index,news.uncertainty_pct.rolling(5,min_periods=3).mean(),color=TEAL,label='Uncertainty-word share')
    axs[1].scatter(news.index,news.negative_pct,color=RED,s=10,alpha=.35)
    axs[1].scatter(news.index,news.uncertainty_pct,color=TEAL,s=10,alpha=.35)
    axs[1].set_title('Dots: available sessions; lines: mean requiring 3 of the last 5 sessions',fontsize=9,loc='left')
    axs[1].set_ylabel('LM words / tokens, %');axs[1].legend(ncol=2,fontsize=8)
    axs[2].plot(gpr.index,gpr.GPRD,color=NAVY,label='Published daily global GPR')
    axs[2].set_ylabel('GPR index');axs[2].legend(fontsize=8)
    for ax in axs:dates(ax);ax.axvline(pd.Timestamp('2026-03-02'),color=GRAY,ls=':',lw=1)
    fig.tight_layout();save(fig,'figure3_news')
    rank=json.loads((OUT/'rank_diagnostics.json').read_text())
    matrix=np.array(rank['standardized_covariance_contrast']);names=['2y','WTI fut.','S&P','10y','DXY','GLD']
    fig,axs=plt.subplots(1,2,figsize=(9,4),gridspec_kw={'width_ratios':[1.2,1]})
    limit=np.max(abs(matrix));im=axs[0].imshow(matrix,cmap='RdBu_r',vmin=-limit,vmax=limit)
    axs[0].set_xticks(range(6),names,rotation=30);axs[0].set_yticks(range(6),names);axs[0].grid(False);axs[0].set_title('Standardized covariance difference',fontsize=11)
    fig.colorbar(im,ax=axs[0],fraction=.046,pad=.04)
    eig=rank['eigenvalues'];axs[1].bar(range(1,7),eig,color=[TEAL if e>=0 else RED for e in eig]);axs[1].axhline(0,color=GRAY,lw=.8)
    axs[1].set_xticks(range(1,7));axs[1].set_xlabel('Ordered eigenvalue');axs[1].set_title('Rank-one restriction is not close',fontsize=11)
    fig.tight_layout();save(fig,'figure4_identification')
    reg=pd.read_csv(OUT/'news_regressions.csv')
    selected=['wti_spot','brent_spot','sp500','efa','eem','gold_gld','dollar']
    short={'wti_spot':'WTI spot','brent_spot':'Brent spot','sp500':'S&P 500','efa':'Developed ex-US','eem':'Emerging markets','gold_gld':'Gold proxy','dollar':'DXY'}
    fig,ax=plt.subplots(figsize=(9,4.6))
    for offset,f,color,label in [(-.12,'d_log_attention',TEAL,'Attention change'),(.12,'d_negative_pct',RED,'Negative-word share change')]:
        q=reg.set_index(['outcome','feature']).loc[[(y,f) for y in selected]].reset_index()
        yy=np.arange(len(q))+offset
        ax.errorbar(q.coefficient,yy,xerr=np.array([q.coefficient-q.ci_low,q.ci_high-q.coefficient]),fmt='o',color=color,label=label,capsize=2,ms=4)
    ax.axvline(0,color=GRAY,lw=1);ax.set_yticks(range(len(selected)),[short[y] for y in selected]);ax.invert_yaxis()
    ax.set_title('News/return associations with 95% HAC intervals',loc='left',fontsize=11)
    ax.set_xlabel('Log-return percentage points per one-SD text change • 95% HAC intervals');ax.legend(ncol=2,loc='lower right',fontsize=8.5)
    fig.tight_layout();save(fig,'figure5_regressions')
    social=pd.read_csv(OUT/'social_monthly.csv',index_col=0)
    fig,ax=plt.subplots(figsize=(9,3.2));ax.bar(np.arange(len(social)),social.candidate_records,color=TEAL)
    ax.set_xticks(range(len(social)),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep*']);ax.set_ylabel('Retrieved keyword-matching records')
    ax.set_title('Truth Social: factual archive activity, not a sentiment measure',loc='left',fontsize=11)
    ax.set_xlabel('2026 • *September 1–16 only; X coverage is not complete')
    fig.tight_layout();save(fig,'figure6_social')


if __name__=='__main__':build()
