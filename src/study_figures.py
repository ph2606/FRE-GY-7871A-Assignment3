"""Observed market trends, event information, and model diagnostics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .figures import style,save,dates,NAVY,TEAL,RED,GOLD,GRAY
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'


def build():
    style()
    from .study_config import MARKET_END
    lev=pd.read_csv(ROOT/'data/processed/market_levels.csv',index_col=0,parse_dates=True).loc['2026-02-27':MARKET_END]
    gsw=pd.read_csv(ROOT/'data/processed/offrun_yields.csv',index_col=0,parse_dates=True).loc['2026-02-27':MARKET_END]
    fig,axes=plt.subplots(3,1,figsize=(9,7.8),sharex=True)
    for col,label,color in [('DCOILWTICO','WTI spot',TEAL),('DCOILBRENTEU','Brent spot',NAVY)]:
        s=lev[col].dropna();axes[0].plot(s.index,s,label=label,color=color)
    axes[0].set_title('Oil prices and the physical Brent–WTI spread',loc='left');axes[0].set_ylabel('USD / barrel');axes[0].legend(ncol=2)
    joint=lev[['DCOILWTICO','DCOILBRENTEU']].dropna()
    axes[1].plot(joint.index,joint.DCOILBRENTEU-joint.DCOILWTICO,color=RED);axes[1].set_ylabel('Brent − WTI\nUSD / barrel')
    axes[2].plot(joint.index,100*(joint.DCOILBRENTEU/joint.DCOILWTICO-1),color=GOLD);axes[2].set_ylabel('Brent premium\n% of WTI')
    for ax in axes:dates(ax);ax.axvline(pd.Timestamp('2026-03-02'),ls=':',color=GRAY)
    fig.tight_layout();save(fig,'study_oil')
    fig,axes=plt.subplots(2,1,figsize=(9,6.3),sharex=True)
    for c,label,color in [('DGS2','2-year CMT',TEAL),('DGS10','10-year CMT',NAVY)]:
        s=lev[c].dropna();axes[0].plot(s.index,s,label=label,color=color)
    for c,label,color in [('SVENPY02','2-year fitted',TEAL),('SVENPY10','10-year fitted',NAVY)]:
        s=gsw[c].dropna();axes[0].plot(s.index,s,label=label,color=color,ls='--',alpha=.7)
    axes[0].set_ylabel('Yield, %');axes[0].set_title('Treasury yields and global financial markets',loc='left');axes[0].legend(ncol=2,fontsize=8)
    for c,label,color in [('^GSPC_close','S&P 500',NAVY),('EFA','EFA',TEAL),('EEM','EEM',GOLD),('GLD','GLD',RED)]:
        s=lev[c].dropna();axes[1].plot(s.index,100*(s/s.iloc[0]-1),label=label,color=color)
    axes[1].set_ylabel('Change from February 27\nprewar close, %');axes[1].legend(ncol=4,fontsize=8)
    for ax in axes:dates(ax)
    fig.tight_layout();save(fig,'study_markets')
    news=pd.read_csv(OUT/'binary_news_days.csv',index_col=0,parse_dates=True)
    fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True,gridspec_kw={'height_ratios':[1,1,.6]})
    regimes=json.loads((OUT/'nlp_regimes_summary.json').read_text())
    axes[0].bar(news.index,news.intensity,color=NAVY,width=1.7,alpha=.7)
    axes[0].axhline(regimes['threshold_articles_per_calendar_day'],ls='--',color=RED,label='Active-window 75th percentile')
    axes[0].set_ylabel('Articles / elapsed day');axes[0].set_title('Primary regime: relevant publisher-news intensity',loc='left',fontsize=10);axes[0].legend(fontsize=8)
    axes[1].bar(news.index,news.gdelt_war_sources,color=TEAL,width=1.7,alpha=.65)
    axes[1].set_ylabel('Source URLs');axes[1].set_title('GDELT robustness: recent Iranian-actor conflict sources',loc='left',fontsize=10)
    axes[2].scatter(news.index,news.high,color=NAVY,s=12)
    axes[2].set_yticks([0,1],['0: low news','1: high news']);axes[2].set_ylim(-.15,1.15)
    axes[2].set_xlabel('2026 US equity session • GDELT uses next-day archive availability')
    for ax in axes:dates(ax);ax.axvline(pd.Timestamp('2026-03-02'),ls=':',color=GRAY)
    fig.tight_layout();save(fig,'study_news')
    rank=json.loads((OUT/'paper_rank.json').read_text());mat=np.array(rank['standardized_covariance_contrast'])
    fig,axes=plt.subplots(1,2,figsize=(9,3.7));bound=np.max(abs(mat));names=['2y','10y','WTI fut.','S&P','DXY','GLD']
    im=axes[0].imshow(mat,cmap='RdBu_r',vmin=-bound,vmax=bound);axes[0].grid(False)
    axes[0].set_xticks(range(6),names,rotation=35);axes[0].set_yticks(range(6),names)
    axes[0].set_title('Standardized second-moment contrast',fontsize=10);fig.colorbar(im,ax=axes[0],fraction=.045)
    eigen=rank['eigenvalues'];axes[1].bar(range(1,7),eigen,color=[TEAL if x>0 else RED for x in eigen]);axes[1].axhline(0,color=GRAY)
    axes[1].set_title('Single-factor restriction: one positive eigenvalue',fontsize=9);axes[1].set_xlabel('Ordered eigenvalue')
    fig.tight_layout();save(fig,'study_identification')
    lp=pd.read_csv(OUT/'duration_local_projections.csv')
    fig,axes=plt.subplots(2,2,figsize=(9,6.5),sharex=True)
    for ax,outcome,label in zip(axes.flat,['wti_spot','brent_spot','sp500','ten_year'],['WTI spot (%)','Brent spot (%)','S&P 500 (%)','10-year CMT (pp)']):
        for feature,color,offset in [('fighting',RED,-.08),('cessation',TEAL,.08)]:
            s=lp.loc[lp.outcome.eq(outcome)&lp.feature.eq(feature)&lp.horizon.ge(0)].sort_values('horizon')
            x=s.horizon.to_numpy()+offset
            ax.errorbar(x,s.coefficient,yerr=np.vstack([s.coefficient-s.ci_low,s.ci_high-s.coefficient]),fmt='o-',color=color,ms=4,lw=1.2,capsize=3,label=feature.capitalize())
        ax.axhline(0,color=GRAY,lw=.8);ax.set_title(label,fontsize=11);ax.set_xticks([0,1,5]);ax.set_xlabel('Forward session horizon')
    axes[0,0].legend(fontsize=8);fig.suptitle('Cumulative market associations per SD of news change; 95% HAC intervals',fontsize=11)
    fig.tight_layout();save(fig,'study_alternative')
    h=pd.read_csv(OUT/'headline_duration_audit.csv');social=json.loads((OUT/'social_timing_summary.json').read_text())
    order=['pre_open','open','after_close','weekend','holiday'];labels=['Before open','Market open','After close','Weekend','Holiday']
    fig,ax=plt.subplots(figsize=(9,3.7));x=np.arange(5)
    a=h.cash_equity_state.value_counts().reindex(order,fill_value=0)/len(h)*100
    b=pd.Series(social['equity_state_counts']).reindex(order,fill_value=0)/social['candidate_records']*100
    ax.bar(x-.18,a,.36,label='Guardian effective news clock',color=NAVY);ax.bar(x+.18,b,.36,label='Truth Social original post clock',color=TEAL)
    ax.set_xticks(x,labels);ax.set_ylabel('Share of retrieved items, %');ax.set_title('Information timing relative to New York cash equities',loc='left');ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'study_timing')


if __name__=='__main__':build()
