"""Iran war-risk study: paper replication first, then duration-news extensions."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from .events import EVENTS
from .paper_replication import replicate_tables
from .heteroskedasticity import covariance_rank_one
from .study_config import WAR_START, MARKET_END
from .news_regimes import build as build_regimes, high_flags

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
PRICE_COLS={'wti_spot':'DCOILWTICO','brent_spot':'DCOILBRENTEU','wti_futures':'CL=F_close',
 'brent_futures':'BZ=F_close','sp500':'^GSPC_close','nasdaq':'^IXIC_close','efa':'EFA','eem':'EEM',
 'gold_gld':'GLD','dollar':'DX-Y.NYB_close','broad_dollar':'DTWEXBGS','vix':'^VIX_close'}
YIELD_COLS={'two_year':'DGS2','ten_year':'DGS10','breakeven_10y':'T10YIE','bbb_spread':'BAMLC0A4CBBB','hy_spread':'BAMLH0A0HYM2'}
LABELS={'two_year':'2-year fitted Treasury','ten_year':'10-year fitted Treasury',
 'breakeven_10y':'10-year breakeven','liquidity':'10-year liquidity premium','sp500':'S&P 500',
 'bbb_spread':'BBB OAS','hy_spread':'High-yield OAS','wti_dollar':'WTI nearby futures',
 'gold_gld':'Gold ETF (GLD)','broad_dollar':'Broad dollar','wti_spot':'WTI spot','brent_spot':'Brent spot',
 'brent_wti_spread':'Brent minus WTI','slope':'10y minus 2y','nasdaq':'Nasdaq','efa':'EFA','eem':'EEM',
 'vix':'VIX','dollar':'DXY','wti_futures':'WTI nearby futures (%)','brent_futures':'Brent nearby futures (%)'}
OUTCOMES=['ten_year','breakeven_10y','liquidity','sp500','bbb_spread','hy_spread','wti_dollar','gold_gld',
 'broad_dollar','wti_spot','brent_spot','brent_wti_spread','slope','nasdaq','efa','eem','vix','dollar','wti_futures','brent_futures']
UNITS={k:'%' for k in PRICE_COLS}|{k:'pp' for k in YIELD_COLS}|{'liquidity':'pp','slope':'pp','wti_dollar':'$/bbl','brent_wti_spread':'$/bbl'}
FOMC=pd.to_datetime(['2026-01-28','2026-03-18','2026-04-29','2026-06-17','2026-07-29','2026-09-16'])
INFO=['Prewar force movement','Prewar force movement','Prewar negotiations','Prewar negotiations','Prewar negotiations',
 'Fighting begins','Energy facilities attacked','Ceasefire announced','Talks inconclusive','Port blockade begins',
 'Reopening announced','Ceasefire extended','Shipping operation announced','Missiles launched','Interim agreement',
 'Sanctions waiver','Ship attacked','Halt in exchanges agreed','Shipping attacks and strikes','Strike pause announced',
 'Talks and vessel attack','Ceasefire period expires','Reopening negotiations','Pipeline disruption','Vessel attack; talks delayed']


def market_panel(curve='gsw'):
    levels=pd.read_csv(ROOT/'data/processed/market_levels.csv',index_col=0,parse_dates=True)
    gsw=pd.read_csv(ROOT/'data/processed/offrun_yields.csv',index_col=0,parse_dates=True)
    equity=levels['^GSPC_close'].dropna().index
    prior=pd.Series(equity,index=equity).shift(1)
    values={k:levels[c].dropna() for k,c in PRICE_COLS.items()}
    values.update({k:levels[c].dropna() for k,c in YIELD_COLS.items()})
    if curve=='gsw':
        values['two_year']=gsw.SVENPY02.dropna();values['ten_year']=gsw.SVENPY10.dropna()
    values['wti_dollar']=levels['CL=F_close'].dropna()
    values['brent_wti_spread']=(levels.DCOILBRENTEU-levels.DCOILWTICO).dropna()
    values['slope']=(values['ten_year']-values['two_year']).dropna()
    panel=pd.DataFrame(index=equity)
    for name,s in values.items():
        last=pd.Series(s.index,index=s.index).shift(1)
        change=100*(s/s.shift(1)-1) if name in PRICE_COLS else s.diff()
        # Equity and each asset's prior observation must be the same session.
        panel[name]=change.reindex(equity).where(last.reindex(equity).eq(prior))
    panel['interval_days']=(pd.Series(equity,index=equity)-prior).dt.days
    panel['brent_wti_spread']=panel.brent_wti_spread.where(panel[['wti_spot','brent_spot']].notna().all(axis=1))
    panel=panel.loc['2026-01-01':MARKET_END]
    return panel,values,levels


def chronology(panel,shift_unknown=False):
    events=pd.DataFrame(EVENTS).copy();events['war_risk']=INFO
    events['event_date']=pd.to_datetime(events.event_date)
    events['market_date']=pd.to_datetime(events.market_date)
    events['weekday']=events.event_date.dt.day_name()
    events['friday_sunday']=events.event_date.dt.weekday.isin([4,5,6])
    events['clock_precision']=np.where(events.event_time_utc.ne(''),'Exact announcement clock',
            np.where(events.event_date.dt.weekday.ge(5),'Weekend date; next session','Date/report clock; intraday uncertain'))
    known={'2026-01-26':'12:19 ET report; open','2026-02-26':'13:42 ET report; open',
           '2026-04-07':'18:32 ET announcement; closed','2026-04-21':'16:22 ET update; closed',
           '2026-08-28':'07:00 ET report; pre-open'}
    events['cash_window']=[known.get(str(d.date()),'Weekend; closed' if d.weekday()>=5 else 'Intraday time uncertain') for d in events.event_date]
    if shift_unknown:
        for i,r in events.iterrows():
            if r.timing_uncertain:
                later=panel.index[panel.index>r.market_date]
                events.at[i,'market_date']=later[0] if len(later) else pd.NaT
    return events


def nearest_controls(panel,events,exclude_adjacent=False):
    """Paper-style nearest non-H controls; no selection on asset changes."""
    panel=panel.copy()
    highs=pd.DatetimeIndex(events.market_date.dropna().unique()).sort_values()
    panel['high']=panel.index.isin(highs)
    eligible=panel.loc[panel.two_year.notna() & panel.sp500.notna()]
    used=set();rows=[];audit=[];blocked=set(highs)
    if exclude_adjacent:
        for high in highs:
            if high in panel.index:
                i=panel.index.get_loc(high);blocked.update(panel.index[max(0,i-1):i+2])
    for high in highs:
        if high not in eligible.index:
            audit.append({'high_date':high,'low_date':pd.NaT,'status':'Reference yield or aligned return unavailable'});continue
        available=[d for d in eligible.index if d not in blocked and d not in used]
        if not available:
            audit.append({'high_date':high,'low_date':pd.NaT,'status':'No unused comparison session'});continue
        low=min(available,key=lambda d:(abs((d-high).days),d>high,d))
        used.add(low);pair_id=len(rows)//2
        for date in [high,low]:rows.append(dict(panel.loc[date],date=date,pair_id=pair_id))
        audit.append({'high_date':high,'low_date':low,'status':'Matched','distance_days':abs((high-low).days)})
    selected=pd.DataFrame(rows).set_index('date').sort_index()
    # Align every outcome to the reference's known interval before deletion.
    full=panel.copy();full.loc[full.two_year.isna(),list(set(OUTCOMES)-{'liquidity'})]=np.nan
    return selected,full,pd.DataFrame(audit)


def estimate(selected,full,events,bootstrap=1999):
    unavailable={'liquidity':'A traded current-issue yield and matching off-the-run curve premium were not acquired.'}
    for name in OUTCOMES:
        if name in unavailable:continue
        valid=selected[['two_year',name]].notna().all(axis=1).groupby(selected.pair_id).all().sum()
        if valid<3:unavailable[name]='Fewer than three complete H/L pairs in this specification.'
    return replicate_tables(selected,OUTCOMES,full_panel=full,events=events,scenario=-.25,reference_unit='pp',
       units=UNITS,labels=LABELS,unavailable=unavailable,
       bootstrap=bootstrap,seed=2603)


def local_projections(panel,values,news):
    """Separate cessation/fighting reporting, post-outbreak, with timing placebos.

    News clocks reflect archive availability. These are retrospective conditional
    associations, not measured war probabilities or causal effects of statements.
    """
    selected=['wti_spot','brent_spot','sp500','two_year','ten_year','gold_gld','dollar','brent_wti_spread']
    news=news.reindex(panel.index)
    features=pd.DataFrame(index=panel.index)
    for name in ['cessation','fighting']:
        s=np.log1p(news[name+'_sources']).diff()
        features[name]=s/s.std(ddof=1)
        features[name+'_lag']=features[name].shift(1)
    rows=[];samples=[]
    for outcome in selected:
        s=values[outcome].reindex(panel.index)
        for horizon in [-1,0,1,5]:
            if horizon==-1:dependent=panel[outcome].shift(1)
            elif outcome in PRICE_COLS:dependent=100*(s.shift(-horizon)/s.shift(1)-1)
            else:dependent=s.shift(-horizon)-s.shift(1)
            data=features.copy();data['y']=dependent
            data['lag_y']=panel[outcome].shift(2 if horizon==-1 else 1)
            data['log_days']=np.log(panel.interval_days)
            data['fomc']=data.index.isin(FOMC).astype(float)
            data=data.loc['2026-03-02':].dropna()
            if len(data)<40:continue
            month=pd.get_dummies(data.index.month,prefix='month',drop_first=True,dtype=float).set_axis(data.index)
            regressors=['cessation','fighting','cessation_lag','fighting_lag','lag_y','log_days','fomc']
            x=sm.add_constant(pd.concat([data[regressors],month],axis=1))
            model=sm.OLS(data.y,x).fit(cov_type='HAC',cov_kwds={'maxlags':5+max(horizon,0),'use_correction':True},use_t=True)
            for feature in ['cessation','fighting']:
                ci=model.conf_int().loc[feature]
                rows.append({'outcome':outcome,'feature':feature,'horizon':horizon,'coefficient':model.params[feature],
                  'se':model.bse[feature],'p_value':model.pvalues[feature],'ci_low':ci.iloc[0],'ci_high':ci.iloc[1],
                  'n':len(data),'unit':UNITS[outcome],'first_date':str(data.index.min().date()),'last_date':str(data.index.max().date())})
            samples.extend({'outcome':outcome,'horizon':horizon,'date':str(d.date())} for d in data.index)
    result=pd.DataFrame(rows)
    result['q_value']=np.nan
    for is_placebo in [False,True]:
        mask=result.horizon.eq(-1).eq(is_placebo)
        result.loc[mask,'q_value']=multipletests(result.loc[mask,'p_value'],method='fdr_bh')[1]
    return result,pd.DataFrame(samples)


def run(bootstrap=1999):
    OUT.mkdir(exist_ok=True)
    regimes,events,regime_summary=build_regimes()
    events['market_date']=pd.to_datetime(events.market_date)
    events['event_date']=pd.to_datetime(events.event_date)
    context,values,levels=market_panel('gsw')
    panel=context.loc[WAR_START:MARKET_END].copy()
    valid=regimes.index[regimes.high.notna()]
    panel=panel.loc[panel.index.intersection(valid)]
    selected,full,audit=nearest_controls(panel,events)
    events['included_in_main_estimation']=events.market_date.isin(selected.index[selected.high])
    result=estimate(selected,full,events,bootstrap)
    events.to_csv(OUT/'paper_table1.csv',index=False)
    for name in ['table2','table3','diagnostics']:
        result[name].to_csv(OUT/('paper_'+name+'.csv'),index=False)
    audit.to_csv(OUT/'paper_matching.csv',index=False)
    full.to_csv(OUT/'paper_full_panel.csv')
    selected.to_csv(OUT/'paper_estimation_panel.csv')
    # Original -25bp and an explicitly separate +25bp yield scenario use the
    # same estimated loadings. No sign is chosen to manufacture predictions.
    rise=result['table2'].copy()
    for method in ['omega1','omega2','pooled']:
        rise[method+'_effect']=-rise[method+'_effect']
        lower=rise[method+'_effect_ci_low'].copy()
        rise[method+'_effect_ci_low']=-rise[method+'_effect_ci_high']
        rise[method+'_effect_ci_high']=-lower
    rise['scenario']=.25
    rise.to_csv(OUT/'iran_yield_rise_scenario.csv',index=False)
    curated=chronology(context)
    curated=curated.loc[curated.event_date.ge(WAR_START)].copy()
    curated.to_csv(OUT/'curated_war_chronology.csv',index=False)
    reactions=[]
    for event in curated.itertuples():
        date=event.market_date;following=context.index[context.index>date]
        next_date=following[0] if len(following) else pd.NaT
        for name in ['wti_spot','brent_spot','sp500','ten_year']:
            reactions.append({'event_date':str(event.event_date.date()),'reaction_session':str(date.date()),
                'following_session':str(next_date.date()) if pd.notna(next_date) else '',
                'friday_sunday':event.friday_sunday,'cash_window':event.cash_window,'outcome':name,
                'reaction_change':context.at[date,name] if date in context.index else np.nan,
                'following_change':context.at[next_date,name] if next_date in context.index else np.nan,
                'unit':UNITS[name],'source_url':event.source_url})
    pd.DataFrame(reactions).to_csv(OUT/'event_reaction_windows.csv',index=False)
    print('NLP main method:',int(selected.high.sum()),'H and',int((~selected.high).sum()),'L sessions',flush=True)
    robust=[]
    def check(name,p,ev,adjacent=False):
        sub,all_days,matching=nearest_controls(p,ev,adjacent)
        if len(sub)<6:return
        rr=estimate(sub,all_days,ev,bootstrap=min(bootstrap,399))['table2']
        rr['specification']=name;robust.append(rr)
    cmt,cmt_values,_=market_panel('cmt')
    check('CMT yields',cmt.reindex(panel.index),events)
    check('Curated event chronology',panel,curated)
    for q in [2/3,.8]:
        flags,threshold=high_flags(regimes.intensity,q,regimes.coverage_complete)
        ev=pd.DataFrame({'market_date':flags.index[flags.eq(1)]})
        check(f'NLP quantile {q:.3f}',panel,ev)
    monthly=pd.Series(np.nan,index=regimes.index)
    for _,group in regimes.groupby(regimes.index.to_period('M')):
        monthly.loc[group.index]=high_flags(group.intensity,.75,group.coverage_complete)[0]
    check('Within-month NLP quantiles',panel,pd.DataFrame({'market_date':monthly.index[monthly.eq(1)]}))
    delayed=regimes.high.shift(1)
    delayed_panel=panel.loc[panel.index.intersection(delayed.dropna().index)]
    check('Next-session NLP timing',delayed_panel,pd.DataFrame({'market_date':delayed.index[delayed.eq(1)]}))
    check('Exclude adjacent controls',panel,events,True)
    news=pd.read_csv(ROOT/'data/processed/duration_daily.csv',index_col=0,parse_dates=True)
    intensity=(news.gdelt_war_sources/regimes.interval_days).reindex(regimes.index)
    gdelt_flags,gdelt_threshold=high_flags(intensity,.75,regimes.coverage_complete)
    check('GDELT Iran-actor NLP',panel,pd.DataFrame({'market_date':gdelt_flags.index[gdelt_flags.eq(1)]}))
    # Exclude the entire pair when either member is a scheduled FOMC date.
    keep=(~selected.index.isin(FOMC))
    ids=selected.loc[keep].groupby('pair_id').size()
    sub=selected.loc[selected.pair_id.isin(ids[ids.eq(2)].index)]
    if len(sub)>=6:
        rr=estimate(sub,full,events,bootstrap=min(bootstrap,399))['table2']
        rr['specification']='Exclude FOMC pairs';robust.append(rr)
    pd.concat(robust,ignore_index=True).to_csv(OUT/'paper_robustness.csv',index=False)
    chosen=['two_year','ten_year','wti_futures','sp500','dollar','gold_gld']
    rank=covariance_rank_one(selected,chosen,regime='high',center=False)
    rank.pop('bootstrap_design',None)
    (OUT/'paper_rank.json').write_text(json.dumps(rank,indent=2,default=str))
    lp,samples=local_projections(cmt,cmt_values,news)
    lp.to_csv(OUT/'duration_local_projections.csv',index=False)
    samples.to_csv(OUT/'duration_lp_samples.csv',index=False)
    # Good/bad/no-retrieved-news is a descriptive extension, with mixed and
    # uncertain days explicit. Market outcomes never determine these labels.
    direction=[]
    for name,group in regimes.groupby('direction_regime',sort=False):
        for outcome in ['two_year','ten_year','wti_spot','brent_spot','sp500','bbb_spread','hy_spread','gold_gld']:
            y=cmt.reindex(group.index)[outcome].dropna()
            direction.append({'regime':name,'sessions':len(group),'outcome':outcome,'observations':len(y),
                'mean_change':y.mean(),'median_change':y.median(),'sd_change':y.std(ddof=1),'unit':UNITS[outcome],
                'interpretation':'Descriptive conditional mean; no causal label or sign validation from market outcomes.'})
    pd.DataFrame(direction).to_csv(OUT/'directional_regime_markets.csv',index=False)
    design=news.reindex(regimes.index).join(regimes[['high','intensity','coverage_complete','direction_regime']])
    design['major_war_news_day']=design.index.isin(curated.market_date).astype(int)
    design['matched_comparison']=design.index.isin(selected.index[~selected.high]).astype(int)
    design['gdelt_high']=gdelt_flags
    design.to_csv(OUT/'binary_news_days.csv')
    # Expected signs are hypotheses, evaluated after unsupervised H selection.
    original={'ten_year':-.26,'breakeven_10y':-.11,'liquidity':-.01,'sp500':-3.76,
              'bbb_spread':.05,'hy_spread':.34,'wti_dollar':.77,'gold_gld':1.30,'broad_dollar':-.44}
    expected={'ten_year':1,'breakeven_10y':1,'sp500':-1,'bbb_spread':1,'hy_spread':1,
              'wti_dollar':1,'wti_spot':1,'brent_spot':1}
    comparison=[]
    for r in rise.itertuples():
        if r.variable not in original and r.variable not in expected:continue
        comparison.append({'variable':r.variable,'label':r.label,'unit_2026':r.unit,
            'original_2003_effect':original.get(r.variable,np.nan),
            'iran_expected_sign':expected.get(r.variable,np.nan),
            'iran_plus25bp_scenario':r.pooled_effect,
            'point_sign_matches_hypothesis':bool(np.sign(r.pooled_effect)==expected[r.variable]) if r.variable in expected and pd.notna(r.pooled_effect) else np.nan,
            'weak_reference_variance_shift':r.weak_reference_variance_shift,
            'original_unit_note':'Original gold is dollars, here GLD percent; original oil is 12-month contract, here nearby.'})
    pd.DataFrame(comparison).to_csv(OUT/'benchmark_expectations.csv',index=False)
    paths=['data/processed/market_levels.csv','data/processed/offrun_yields.csv','data/processed/duration_daily.csv',
           'data/processed/gdelt_iran_events.csv','data/processed/social_candidates.csv',
           'data/processed/news_features.csv','outputs/nlp_news_regimes.csv']
    diag={'start':WAR_START,'cutoff':MARKET_END,'created_utc':datetime.now(timezone.utc).isoformat(),
      'main_regime':'Unsigned NLP high-intensity Guardian war coverage, full active-window strict upper-quartile threshold',
      'main_curve':'Federal Reserve GSW fitted off-the-run coupon-equivalent par yields',
      'curve_last_date':str(values['two_year'].index.max().date()),'event_records':len(events),
      'event_sessions':int(events.market_date.nunique()),'main_pairs':int(selected.high.sum()),
      'low_news_sessions':regime_summary['low_sessions'],'equity_sessions':len(regimes),
      'no_major_event_sessions':int(design.major_war_news_day.eq(0).sum()),
      'main_weak_rows':int(result['diagnostics'].weak_reference_variance_shift.sum()),
      'main_available_outcomes':len(result['diagnostics']),'rank_one_residual':rank['rank_one_residual'],
      'lp_primary_tests':int(lp.horizon.ge(0).sum()),'lp_primary_q_lt05':int(((lp.q_value<.05)&lp.horizon.ge(0)).sum()),
      'lp_placebo_q_lt05':int(((lp.q_value<.05)&lp.horizon.eq(-1)).sum()),
      'lp_min_primary_q':float(lp.loc[lp.horizon.ge(0),'q_value'].min()),
      'nlp_regimes':regime_summary,'gdelt_threshold':gdelt_threshold,
      'gdelt_high_overlap_sessions':int((gdelt_flags.eq(1)&regimes.high.eq(1)).sum()),
      'matching_max_calendar_gap':int(audit.distance_days.max()),
      'input_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
      'method':result['metadata']}
    (OUT/'study_diagnostics.json').write_text(json.dumps(diag,indent=2))
    print(json.dumps({k:v for k,v in diag.items() if k not in ['method','input_sha256']},indent=2))
    return diag


if __name__=='__main__':run()
