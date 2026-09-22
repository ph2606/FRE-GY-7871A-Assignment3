"""Select high-news regimes from text, without observing market outcomes.

Full-window quantiles are a retrospective research design, not forecasts.
Low coverage is not a claim that the day was driven only by fundamentals.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .study_config import WAR_START, MARKET_END

ROOT=Path(__file__).resolve().parents[1]
ZONE='America/New_York'


def high_flags(intensity, quantile=.75, complete=None):
    if not 0 < quantile < 1:raise ValueError('Quantile must lie strictly between zero and one')
    values=pd.to_numeric(intensity,errors='raise')
    if (values.dropna()<0).any():raise ValueError('News intensity cannot be negative')
    if complete is not None:values=values.where(complete)
    threshold=values.quantile(quantile)
    result=values.gt(threshold).astype(float).where(values.notna())
    return result,float(threshold)


def build_regimes(headlines, sessions, documents=None, quantile=.75):
    sessions=pd.DatetimeIndex(sessions).sort_values().unique()
    active=sessions[(sessions>=WAR_START)&(sessions<=MARKET_END)]
    if not len(active):raise ValueError('No observed active-conflict sessions')
    end_clocks=pd.Series([pd.Timestamp(d).tz_localize(ZONE)+pd.Timedelta(hours=16) for d in sessions],index=sessions)
    prior=end_clocks.shift(1).reindex(active)
    start=pd.Timestamp(WAR_START,tz=ZONE)
    prior=prior.map(lambda x:max(x,start) if pd.notna(x) else start)
    frame=pd.DataFrame(index=active);frame.index.name='date'
    frame['interval_days']=(end_clocks.reindex(active)-prior).dt.total_seconds()/86400
    h=headlines.copy()
    h['clock']=pd.to_datetime(h.effective_utc,utc=True)
    h['session']=pd.to_datetime(h.reaction_close_session)
    h['relevant']=h.war_relevant.astype(str).str.lower().eq('true')
    h=h.loc[h.clock.ge(start) & h.session.isin(active)].drop_duplicates('url').copy()
    war=h.loc[h.relevant]
    frame['war_articles']=war.groupby('session').size().reindex(active,fill_value=0)
    frame['articles']=h.groupby('session').size().reindex(active,fill_value=0)
    frame['friday_sunday_articles']=war.groupby('session').is_friday_to_sunday.sum().reindex(active,fill_value=0)
    frame['coverage_complete']=True
    missing=[]
    if documents is not None:
        # A failed page has no reliable intraday clock. Conservatively exclude
        # its dated session and the next session from regime assignment.
        failed=documents.loc[documents.exclusion.fillna('').str.startswith('download/parse failure')]
        for date in pd.to_datetime(failed.url_date,errors='coerce').dropna():
            candidates=active[active>=date][:2]
            for d in candidates:
                frame.at[d,'coverage_complete']=False;missing.append(str(d.date()))
    frame['intensity']=frame.war_articles/frame.interval_days
    frame['high'],threshold=high_flags(frame.intensity,quantile,frame.coverage_complete)
    frame['any_war_news']=frame.war_articles.gt(0).astype(float).where(frame.coverage_complete)
    categories={'cessation':'Cessation/reopening language','continuation':'Continuing fighting/disruption language',
                'mixed':'Mixed physical developments','attributed':'Attributed statement/policy context',
                'negotiation':'Negotiation context without verified cessation','unclear':'War context; duration unclear'}
    for name,category in categories.items():
        frame[name+'_articles']=war.category.eq(category).groupby(war.session).sum().reindex(active,fill_value=0)
    c=frame.cessation_articles.gt(0);f=frame.continuation_articles.gt(0);mixed=frame.mixed_articles.gt(0)
    frame['direction_regime']=np.select([
        ~frame.coverage_complete,frame.war_articles.eq(0),mixed|(c&f),c&~f,f&~c],
        ['Coverage incomplete','No retrieved war news','Mixed physical language','Good: cessation language','Bad: continuation language'],
        default='Direction unclear / attributed claims')
    # Attributed or uncertain stories can coexist with a physical-language flag.
    # They are not silently converted to verified good/bad physical events.
    frame['qualitative_context_articles']=frame[['attributed_articles','negotiation_articles','unclear_articles']].sum(axis=1)
    calendar=[]
    for date,row in frame.loc[frame.high.eq(1)].iterrows():
        source=war.loc[war.session.eq(date)].sort_values(['clock','url']).iloc[0]
        calendar.append({'event_date':str(source.clock.tz_convert(ZONE).date()),'market_date':str(date.date()),
            'description':source.title,'source_url':source.url,'war_risk':'Direction not needed for variance identification',
            'weekday':source.clock.tz_convert(ZONE).day_name(),'friday_sunday':bool(source.is_friday_to_sunday),
            'cash_window':source.cash_equity_state,'war_articles':int(row.war_articles),
            'intensity':row.intensity,'threshold':threshold,'interval_days':row.interval_days,
            'friday_sunday_articles':int(row.friday_sunday_articles),
            'source_selection':'Earliest effective-clock relevant headline in the high-news window; illustrative, not the sole event.'})
    summary={'start':WAR_START,'end':MARKET_END,'first_session':str(active[0].date()),'last_session':str(active[-1].date()),
        'sessions':len(frame),'high_sessions':int(frame.high.eq(1).sum()),'low_sessions':int(frame.high.eq(0).sum()),
        'missing_coverage_sessions':int(frame.high.isna().sum()),'missing_coverage_dates':sorted(set(missing)),
        'war_articles_mapped':len(war),'quantile':quantile,'threshold_articles_per_calendar_day':threshold,
        'rule':'H=1 if relevant articles per calendar day strictly exceeds the active-window 75th percentile; H=0 otherwise; missing coverage is NA.',
        'selection_uses_market_returns':False,'retrospective_threshold':True,
        'direction_counts':frame.direction_regime.value_counts().to_dict(),
        'direction_limit':'Literal physical headline language, not verified events or calibrated end probabilities. Mixed and unclear states remain separate.'}
    return frame,pd.DataFrame(calendar),summary


def build():
    levels=pd.read_csv(ROOT/'data/processed/market_levels.csv',index_col=0,parse_dates=True)
    headlines=pd.read_csv(ROOT/'outputs/headline_duration_audit.csv')
    documents=pd.read_csv(ROOT/'data/processed/news_documents.csv')
    frame,calendar,summary=build_regimes(headlines,levels['^GSPC_close'].dropna().index,documents)
    frame.to_csv(ROOT/'outputs/nlp_news_regimes.csv')
    calendar.to_csv(ROOT/'outputs/nlp_high_news_chronology.csv',index=False)
    frame.assign(month=frame.index.strftime('%Y-%m')).groupby('month').agg(sessions=('high','size'),
        high=('high','sum'),war_articles=('war_articles','sum'),mean_intensity=('intensity','mean')).to_csv(ROOT/'outputs/nlp_regimes_monthly.csv')
    (ROOT/'outputs/nlp_regimes_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return frame,calendar,summary


if __name__=='__main__':
    *_,summary=build();print(json.dumps(summary,indent=2))
