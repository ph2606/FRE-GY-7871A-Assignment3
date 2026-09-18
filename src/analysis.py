"""Event covariance contrasts and separate descriptive news/market regressions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from .market import SPECS
from .events import EVENTS
from .heteroskedasticity import estimate_pair,covariance_rank_one

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
LABELS={k:label for k,col,unit,label in SPECS}|{'brent_wti_spread':'Brent minus WTI spot spread'}
UNITS={k:unit for k,col,unit,label in SPECS}|{'brent_wti_spread':'USD_per_barrel'}
OUTCOMES=['wti_spot','brent_spot','brent_wti_spread','sp500','nasdaq','efa','eem','vix','gold_gld',
          'dollar','broad_dollar','two_year','ten_year','breakeven_10y','bbb_spread','hy_spread',
          'wti_futures','brent_futures']
FOMC=pd.DatetimeIndex(['2026-01-28','2026-03-18','2026-04-29','2026-06-17','2026-07-29','2026-09-16'])


def load():
    market=pd.read_csv(ROOT/'data/processed/market_changes.csv',index_col=0,parse_dates=True)
    news=pd.read_csv(ROOT/'data/processed/news_daily.csv',index_col=0,parse_dates=True)
    levels=pd.read_csv(ROOT/'data/processed/market_levels.csv',index_col=0,parse_dates=True)
    for c in market.columns:
        if c.endswith('_prior_date'):market[c]=pd.to_datetime(market[c])
    panel=market.join(news).loc['2026-01-01':'2026-09-16'].copy()
    return panel,levels,news


def aligned(panel,outcomes,anchor='sp500'):
    mask=panel[anchor].notna()
    for y in outcomes:
        mask &= panel[y].notna() & panel[y+'_prior_date'].eq(panel[anchor+'_prior_date'])
    return mask


def event_calendar(panel,shift_unknown=False):
    sessions=panel.index[panel.sp500.notna()]
    rows=[]
    for event in EVENTS:
        row=event.copy();date=pd.Timestamp(row['market_date'])
        uncertain=str(row['timing_uncertain']).lower()=='true'
        if shift_unknown and uncertain:
            following=sessions[sessions>date]
            date=following[0] if len(following) else pd.NaT
        row['market_date']=date;rows.append(row)
    return pd.DataFrame(rows)


def make_pairs(panel,kind='events',shift_unknown=False):
    """Choose dates using source metadata/news counts, never return magnitudes.

    Low controls: within 35 calendar days, same pre/post February-28 period,
    not an event or adjacent trading session, at/below their month's median
    attention. Prefer same-month then nearest date. No replacement.
    """
    sessions=panel.index[panel.sp500.notna()]
    eligible=panel.loc[aligned(panel,['two_year']) & panel.war_attention.notna()].copy()
    events=event_calendar(panel,shift_unknown)
    event_dates=pd.DatetimeIndex(events.market_date.dropna().unique()).sort_values()
    if kind=='events':
        highs=event_dates
    else:
        highs=[]
        for _,g in eligible.groupby(eligible.index.to_period('M')):
            n=max(1,len(g)//4)
            # Stable chronological ordering resolves equal counts reproducibly.
            highs.extend(g.sort_values('war_attention',ascending=False,kind='stable').head(n).index)
        highs=pd.DatetimeIndex(sorted(highs))
    blocked=set(event_dates)|set(highs)
    for date in highs:
        if date in sessions:
            i=sessions.get_loc(date)
            blocked.update(sessions[max(0,i-1):i+2])
    med=eligible.groupby(eligible.index.to_period('M')).war_attention.transform('median')
    low=eligible.loc[(eligible.war_attention<=med) & ~eligible.index.isin(blocked)]
    used=set();pairs=[];audit=[]
    for high in highs:
        if high not in eligible.index:
            audit.append({'high_date':high,'status':'no aligned two-year/equity observation'});continue
        candidates=[]
        for date in low.index:
            if date in used or abs((date-high).days)>35:continue
            if (date<pd.Timestamp('2026-02-28'))!=(high<pd.Timestamp('2026-02-28')):continue
            if kind!='events' and low.at[date,'war_attention']>=eligible.at[high,'war_attention']:continue
            candidates.append(date)
        if not candidates:
            audit.append({'high_date':high,'status':'no unused lower-attention control within 35 days'});continue
        lowdate=min(candidates,key=lambda d:(d.month!=high.month,abs((d-high).days),d>high,d))
        used.add(lowdate);pair_id=len(pairs)//2
        for date,flag in [(high,True),(lowdate,False)]:
            row=panel.loc[date].to_dict();row.update(date=date,high=flag,pair_id=pair_id)
            pairs.append(row)
        audit.append({'pair_id':pair_id,'high_date':high,'low_date':lowdate,'status':'matched',
                      'high_attention':eligible.at[high,'war_attention'],'low_attention':eligible.at[lowdate,'war_attention'],
                      'gap_calendar_days':abs((high-lowdate).days)})
    return pd.DataFrame(pairs),pd.DataFrame(audit)


def model_sample(pairs,y,anchor):
    frame=pairs.copy()
    match=frame[y+'_prior_date'].eq(frame[anchor+'_prior_date']) & frame[anchor+'_prior_date'].eq(frame.sp500_prior_date)
    frame.loc[~match,[y,anchor]]=np.nan
    # The estimator drops both observations if either member is incomplete.
    return frame


def estimate_set(panel,pairs,anchor='two_year',bootstrap=999,center=True,name='event_centered'):
    rows=[]
    for j,y in enumerate(OUTCOMES):
        if y==anchor:continue
        frame=model_sample(pairs,y,anchor)
        try:r=estimate_pair(frame,x=anchor,y=y,bootstrap=bootstrap,seed=2603+j,center=center)
        except ValueError as e:
            rows.append({'y':y,'x':anchor,'specification':name,'error':str(e)});continue
        r['specification']=name;r['unit']=UNITS[y]
        normalization=-25 if anchor=='two_year' else 100*np.log(1.10)
        r['normalization']=normalization
        for method in ['d_reference','d_outcome','d_pooled']:
            r[method+'_scenario']=normalization*r[method]
            bounds=[normalization*r[method+'_ci_low'],normalization*r[method+'_ci_high']]
            r[method+'_scenario_ci_low']=min(bounds);r[method+'_scenario_ci_high']=max(bounds)
        # Original-style conditional variance quantities; never clipped to [0,100].
        full=panel.loc[aligned(panel,[anchor,y]),y].dropna()
        var_all=float(((full-full.mean())**2).mean() if center else (full**2).mean())
        q=r['d_pooled']**2*r['delta_var_x']
        r.update(incremental_variance=q,high_variance_share=100*q/r['var_y_high'],
                 all_variance_share=100*r['n_high']*q/(len(full)*var_all),all_sample_n=len(full),
                 variance_attribution_caution=bool(r['weak_identification'] or q<0 or q>r['var_y_high']))
        rows.append(r)
    return pd.DataFrame(rows)


def regressions(panel,news,version='primary'):
    work=panel.copy()
    features=['d_log_attention','d_negative_pct']
    if version=='uncertainty':features=['d_log_attention','d_uncertainty_pct']
    if version=='publication_clock':
        ix=work.index[work.sp500.notna()]
        work.loc[ix,'d_log_attention']=np.log1p(work.loc[ix,'war_articles_publication']/work.loc[ix,'interval_days']).diff()
        work.loc[ix,'d_negative_pct']=work.loc[ix,'negative_pct_publication'].diff()
    if version=='next_session':
        # Shift on actual observed equity sessions, not on weekend rows.
        ix=work.index[work.sp500.notna()]
        work.loc[ix,features]=work.loc[ix,features].shift(1)
    scales={c:float(work.loc[work.sp500.notna(),c].std(ddof=1)) for c in features}
    rows=[];samples=[]
    for y in OUTCOMES:
        sub=work.loc[work.sp500.notna()].copy()
        sub['lag_y']=sub[y].shift(1)
        for c in features:sub[c]=sub[c]/scales[c]
        sub['log_days']=np.log(sub.interval_days)
        interval_ok=aligned(sub,[y])
        # The lag is the preceding equity-session return, so both its own
        # observation interval and the current outcome interval must agree.
        mask=interval_ok & interval_ok.shift(1,fill_value=False)
        if version=='exclude_fomc':mask &= ~sub.index.isin(FOMC)
        sub=sub.loc[mask].dropna(subset=[y,'lag_y','log_days']+features)
        if len(sub)<40:continue
        month=pd.get_dummies(sub.index.month,prefix='month',drop_first=True,dtype=float).set_axis(sub.index)
        x=sm.add_constant(pd.concat([sub[features+['lag_y','log_days']],month],axis=1)).astype(float)
        fit=sm.OLS(sub[y].astype(float),x).fit(cov_type='HAC',cov_kwds={'maxlags':5,'use_correction':True},use_t=True)
        for f in features:
            ci=fit.conf_int().loc[f]
            rows.append({'outcome':y,'feature':f,'version':version,'coefficient':fit.params[f],
                         'se':fit.bse[f],'p_value':fit.pvalues[f],'ci_low':ci.iloc[0],'ci_high':ci.iloc[1],
                         'n':len(sub),'r_squared':fit.rsquared,'feature_sd':scales[f],
                         'first':str(sub.index.min().date()),'last':str(sub.index.max().date())})
        samples.extend({'version':version,'outcome':y,'date':str(d.date())} for d in sub.index)
    result=pd.DataFrame(rows)
    result['q_value']=multipletests(result.p_value,method='fdr_bh')[1]
    return result,pd.DataFrame(samples)


def summarize(panel,levels,news):
    rows=[]
    for name,col,unit,label in SPECS:
        if name not in OUTCOMES:continue
        s=levels.loc['2026-01-01':,col].dropna();changes=panel[name].dropna()
        start=s.iloc[0];end=s.iloc[-1]
        change=100*np.log(end/start) if unit=='log_percent' else 100*(end-start)
        rows.append({'variable':name,'label':label,'first_date':str(s.index[0].date()),'last_date':str(s.index[-1].date()),
            'first':start,'last':end,'change':change,'unit':unit,'n':len(changes),'daily_sd':changes.std(),
            'min':s.min(),'max':s.max()})
    s=levels.loc['2026-01-01':,'brent_wti_spread'].dropna()
    rows.append({'variable':'brent_wti_spread','label':LABELS['brent_wti_spread'],'first_date':str(s.index[0].date()),
       'last_date':str(s.index[-1].date()),'first':s.iloc[0],'last':s.iloc[-1],'change':s.iloc[-1]-s.iloc[0],
       'unit':'USD_per_barrel','n':panel.brent_wti_spread.notna().sum(),'daily_sd':panel.brent_wti_spread.std(),'min':s.min(),'max':s.max()})
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'market_summary.csv',index=False)
    monthly=news.groupby(news.index.to_period('M')).agg(articles=('articles','sum'),war_articles=('war_articles','sum'),
        sessions=('articles','size'),attention=('war_attention','mean'),negative_pct=('negative_pct','mean'),
        uncertainty_pct=('uncertainty_pct','mean'),threat_pct=('threat_pct','mean'),act_pct=('act_pct','mean'))
    monthly.to_csv(OUT/'news_monthly.csv')
    trend=[]
    for column in ['log_attention','negative_pct','uncertainty_pct','threat_pct','act_pct']:
        d=news[[column]].dropna().copy()
        d['time_months']=(d.index-pd.Timestamp('2026-01-01')).days/30.4375
        d['active_conflict']=(d.index>=pd.Timestamp('2026-03-02')).astype(float)
        fit=sm.OLS(d[column],sm.add_constant(d[['time_months','active_conflict']])).fit(cov_type='HAC',cov_kwds={'maxlags':5,'use_correction':True},use_t=True)
        for term in ['time_months','active_conflict']:
            trend.append({'feature':column,'term':term,'coefficient':fit.params[term],'p_value':fit.pvalues[term],
                          'ci_low':fit.conf_int().loc[term].iloc[0],'ci_high':fit.conf_int().loc[term].iloc[1],'n':len(d)})
    tr=pd.DataFrame(trend);tr['q_value']=multipletests(tr.p_value,method='fdr_bh')[1];tr.to_csv(OUT/'trend_tests.csv',index=False)
    return summary,monthly


def run(bootstrap=1999):
    OUT.mkdir(parents=True,exist_ok=True)
    panel,levels,news=load()
    events=event_calendar(panel);events.to_csv(OUT/'event_calendar.csv',index=False)
    pairs,audit=make_pairs(panel);pairs.to_csv(OUT/'event_pairs.csv',index=False);audit.to_csv(OUT/'matching_audit.csv',index=False)
    print('Event pairs',len(pairs)//2,flush=True)
    main=estimate_set(panel,pairs,bootstrap=bootstrap)
    main.to_csv(OUT/'heteroskedasticity.csv',index=False)
    oil=estimate_set(panel,pairs,anchor='wti_futures',bootstrap=bootstrap,name='event_oil_anchor')
    oil.to_csv(OUT/'oil_anchor.csv',index=False)
    robust=[estimate_set(panel,pairs,bootstrap=399,center=False,name='uncentered_second_moments')]
    exclude_ids=pairs.loc[pairs.date.isin(FOMC),'pair_id']
    robust.append(estimate_set(panel,pairs.loc[~pairs.pair_id.isin(exclude_ids)],bootstrap=399,name='exclude_fomc_pairs'))
    shifted,shift_audit=make_pairs(panel,shift_unknown=True)
    shifted.to_csv(OUT/'shifted_event_pairs.csv',index=False)
    robust.append(estimate_set(panel,shifted,bootstrap=399,name='shift_uncertain_events'))
    nlp_pairs,nlp_audit=make_pairs(panel,kind='nlp')
    nlp_pairs.to_csv(OUT/'nlp_pairs.csv',index=False);nlp_audit.to_csv(OUT/'nlp_matching_audit.csv',index=False)
    robust.append(estimate_set(panel,nlp_pairs,bootstrap=399,name='within_month_news_regimes'))
    for name,start,end in [('january_may','2026-01-01','2026-05-31'),('june_september','2026-06-01','2026-09-16')]:
        keep=pairs.groupby('pair_id').date.apply(lambda s:s.between(start,end).all())
        sub=pairs.loc[pairs.pair_id.isin(keep[keep].index)]
        robust.append(estimate_set(panel,sub,bootstrap=399,name=name))
    pd.concat(robust,ignore_index=True).to_csv(OUT/'heteroskedasticity_robustness.csv',index=False)
    # Leave-one-pair-out point estimates reveal event dominance without p-hacking.
    influence=[]
    for pid in pairs.pair_id.unique():
        hdate=pairs.loc[(pairs.pair_id==pid)&pairs.high,'date'].iloc[0]
        for y in ['wti_spot','brent_spot','brent_wti_spread','sp500','ten_year']:
            try:
                r=estimate_pair(model_sample(pairs.loc[pairs.pair_id!=pid],y,'two_year'),'two_year',y,bootstrap=0)
                influence.append({'excluded_pair':pid,'excluded_high_date':hdate,'outcome':y,'d_pooled':r['d_pooled'],
                                  'delta_var_x':r['delta_var_x'],'n_pairs':r['n_pairs']})
            except ValueError:pass
    pd.DataFrame(influence).to_csv(OUT/'event_influence.csv',index=False)
    rankcols=['two_year','wti_futures','sp500','ten_year','dollar','gold_gld']
    rankframe=pairs.copy()
    for y in rankcols:
        rankframe.loc[~rankframe[y+'_prior_date'].eq(rankframe.sp500_prior_date),y]=np.nan
    rank=covariance_rank_one(rankframe,rankcols)
    (OUT/'rank_diagnostics.json').write_text(json.dumps(rank,indent=2))
    regs=[];samples=[]
    for version in ['primary','uncertainty','publication_clock','next_session','exclude_fomc']:
        r,s=regressions(panel,news,version);regs.append(r);samples.append(s)
    pd.concat(regs,ignore_index=True).to_csv(OUT/'news_regressions_all.csv',index=False)
    regs[0].to_csv(OUT/'news_regressions.csv',index=False)
    pd.concat(samples,ignore_index=True).to_csv(OUT/'regression_samples.csv',index=False)
    summary,monthly=summarize(panel,levels,news)
    gpr=pd.read_csv(ROOT/'data/processed/gpr_benchmark.csv',index_col=0,parse_dates=True)
    benchmark=news[['log_attention','negative_pct']].join(gpr[['GPRD','GPRD_ACT','GPRD_THREAT']]).dropna()
    benchmark.corr().to_csv(OUT/'benchmark_correlations.csv')
    # Selected dates are audit illustrations, not evidence a post caused a return.
    panel.loc[panel.index.isin(pd.to_datetime(events.market_date)),OUTCOMES+['war_articles','war_attention','negative_pct']].to_csv(OUT/'event_market_changes.csv')
    paths=[ROOT/'data/processed'/p for p in ['market_changes.csv','market_levels.csv','news_features.csv','news_daily.csv','gpr_benchmark.csv','social_candidates.csv']]
    diagnostics={'created_utc':datetime.now(timezone.utc).isoformat(),'cutoff':'2026-09-16','event_records':len(events),
      'unique_event_dates':events.market_date.nunique(),'matched_pairs':len(pairs)//2,'nlp_pairs':len(nlp_pairs)//2,
      'primary_news_tests':len(regs[0]),'primary_news_unadjusted_p_lt_05':int(regs[0].p_value.lt(.05).sum()),
      'primary_news_q_lt_05':int(regs[0].q_value.lt(.05).sum()),'bootstrap_repetitions':bootstrap,
      'weak_two_year_pairs':int(main.weak_identification.sum()),'two_year_estimates':len(main),
      'benchmark_n':len(benchmark),'rank_one_residual':rank['rank_one_residual'],
      'input_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
      'limitations':['Retrospective hand-selected event calendar; not preregistered or exhaustive.',
      'Only war shock variance may change for structural interpretation; daily macro and oil-supply shocks can violate this.',
      'Paired bootstrap assumes independent pairs, not a serial-dependence-robust rank test.',
      'News bodies are revised snapshots; conservative last-modified clock is not an archived real-time information set.',
      'No calibrated war probability, causal effect of posts, or profitable trading backtest is claimed.']}
    (OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print(json.dumps({k:v for k,v in diagnostics.items() if k not in ['input_sha256','limitations']},indent=2))
    print(main[['y','n_pairs','var_ratio','d_pooled_scenario','weak_identification']].round(4).to_string(index=False))
    return diagnostics


if __name__=='__main__':run()
