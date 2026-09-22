"""Duration-relevant physical news categories, distinct from sentiment ratings.

Every retrieved item receives a nominal category. Unclear, mixed, quoted and
official/policy statements are not forced into a numerical war-end probability.
Binary variables describe news/event presence, never an official's performance.
"""
from pathlib import Path
import json,re
from urllib.parse import urlsplit,urlunsplit,parse_qsl,urlencode
import numpy as np
import pandas as pd
from .social_timing import nyse_schedule_2026,timing_fields
from .text_features import IRAN,CONFLICT,ATTRIBUTION

ROOT=Path(__file__).resolve().parents[1]
CESSATION=re.compile(r'\b(?:ceasefire|truce|(?:war|fighting|hostilities) (?:ends?|halts?|stops?)|'
    r'(?:shipping|traffic|exports?) (?:resumes?|restarts?)|reopen\w*|blockade (?:lifted|eased)|'
    r'withdrawal|withdraws|attacks? (?:halted|suspended)|strikes? (?:halted|suspended))\b',re.I)
CONTINUATION=re.compile(r'\b(?:attacks?|strikes?|struck|missiles?|bomb\w*|drones?|'
    r'blockade|closed|closure|damaged|destroyed|killed|hostilities|fighting|'
    r'(?:war|conflict) (?:continues?|extends?|drags))\b',re.I)
NEGATED_END=re.compile(r'\b(?:(?:no|without|reject\w*)\s+(?:a\s+)?ceasefire|'
    r'(?:ceasefire|truce).{0,25}(?:expires?|ends?|collaps\w*|breaks?|violat\w*|fragile))\b',re.I)
TALKS=re.compile(r'\b(?:talks?|negotiat\w*|agreement|deal|mediator\w*|diploma\w*|proposal)\b',re.I)
OFFICIAL=re.compile(r'\b(?:trump|netanyahu|khamenei|araghchi|rubio|hegseth|vance|'
    r'president|minister|government|white house|pentagon|senator|official|'
    r'sanction\w*|tariff\w*|policy|policies|propos\w*)\b',re.I)


def headline_category(title,relevant):
    """Literal headline context; categorical evidence, not calibrated forecasts."""
    title=str(title)
    if not relevant:return 'Outside Iran-war scope'
    # Attributed political claims stay visible qualitatively, without ratings.
    if OFFICIAL.search(title) or ATTRIBUTION.search(title) or any(q in title for q in ['“','”','"']):
        return 'Attributed statement/policy context'
    ending=bool(CESSATION.search(title)) and not bool(NEGATED_END.search(title))
    # Do not call a lifted blockade or halted attack continued fighting solely
    # because the object noun remains in the sentence.
    reduced=re.sub(r'blockade (?:lifted|eased)|attacks? (?:halted|suspended)|strikes? (?:halted|suspended)', '', title,flags=re.I)
    fighting=bool(CONTINUATION.search(reduced)) or bool(NEGATED_END.search(title))
    if ending and fighting:return 'Mixed physical developments'
    if ending:return 'Cessation/reopening language'
    if fighting:return 'Continuing fighting/disruption language'
    if TALKS.search(title):return 'Negotiation context without verified cessation'
    return 'War context; duration unclear'


def canonical_url(url):
    parts=urlsplit(str(url))
    query=[(k,v) for k,v in parse_qsl(parts.query,keep_blank_values=True)
           if not k.lower().startswith('utm_') and k.lower() not in ['fbclid','gclid']]
    return urlunsplit((parts.scheme.lower(),parts.netloc.lower(),parts.path.rstrip('/'),urlencode(sorted(query)),''))


def build():
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    levels=pd.read_csv(ROOT/'data/processed/market_levels.csv',index_col=0,parse_dates=True)
    observed=levels['^GSPC_close'].dropna().loc['2026-01-01':'2026-09-18'].index
    schedule=nyse_schedule_2026()
    docs=pd.read_csv(ROOT/'data/processed/news_features.csv').fillna('')
    rows=[]
    for doc in docs.itertuples():
        stamp=pd.Timestamp(doc.effective_utc)
        timing=timing_fields(stamp,schedule,observed)
        title=doc.title or doc.archive_title
        relevant=str(doc.war_relevant).lower()=='true'
        rows.append({'url':doc.url,'title':title,'provider':'Guardian','text_basis':'Headline; body-based relevance',
             'war_relevant':relevant,'category':headline_category(title,relevant),
             'published_utc':doc.published_utc,'effective_utc':doc.effective_utc,**timing})
    headlines=pd.DataFrame(rows)
    headlines.to_csv(out/'headline_duration_audit.csv',index=False)
    bulk=pd.read_csv(ROOT/'data/processed/gdelt_iran_events.csv',dtype={'event_code':str,'event_root':str,'event_id':str})
    bulk['url_key']=bulk.url.map(canonical_url)
    bulk['recent_event']=bulk.recent_event.astype(str).str.lower().eq('true')
    for col in ['cessation_code','fighting_code','war_event_record']:
        bulk[col]=bulk[col].astype(str).str.lower().eq('true')
    bulk['iran_actor']=bulk.actor1_country.eq('IRN') | bulk.actor2_country.eq('IRN')
    source=bulk.groupby('url_key',sort=True).agg(url=('url','first'),first_archive_date=('archive_date','min'),
          last_archive_date=('archive_date','max'),event_records=('event_id','size'),
          cessation_report=('cessation_code','any'),fighting_report=('fighting_code','any'),
          recent_war_record=('war_event_record','any'),iran_actor=('iran_actor','any'))
    source['category']=np.select([source.cessation_report & source.fighting_report,source.cessation_report,source.fighting_report],
          ['Mixed cessation and fighting codes','Cessation/truce/withdrawal code','Fighting/violence code'],default='Other Iran-related code')
    source['text_basis']='GDELT CAMEO event codes; source article text not returned'
    source.to_csv(out/'gdelt_source_classification.csv',index=False)
    # One count per source URL and availability session, not one per repeated
    # actor-event record. Older-event mentions stay in the audit, not the signal.
    # Action-geography-only matches can include unrelated events. Require an
    # Iran-coded actor for the empirical signal; retain broader rows in audits.
    active=bulk.loc[bulk.iran_actor & bulk.recent_event & (bulk.cessation_code|bulk.fighting_code)].copy()
    mapping={stamp:timing_fields(pd.Timestamp(stamp),schedule,observed)['reaction_close_session']
             for stamp in active.conservative_available_et.unique()}
    active['session']=pd.to_datetime(active.conservative_available_et.map(mapping))
    active=active.groupby(['session','url_key']).agg(cessation=('cessation_code','any'),fighting=('fighting_code','any')).reset_index()
    daily=pd.DataFrame(index=observed);daily.index.name='date'
    for name in ['cessation','fighting']:
        daily[name+'_sources']=active.groupby('session')[name].sum().reindex(observed,fill_value=0)
        daily[name+'_news']=daily[name+'_sources'].gt(0).astype(int)
        daily['log_'+name+'_sources']=np.log1p(daily[name+'_sources'])
    daily['gdelt_war_sources']=active.groupby('session').size().reindex(observed,fill_value=0)
    daily['any_gdelt_war_news']=daily.gdelt_war_sources.gt(0).astype(int)
    day=pd.to_datetime(headlines.reaction_close_session)
    daily['guardian_war_articles']=headlines.war_relevant.groupby(day).sum().reindex(observed,fill_value=0)
    daily['any_guardian_war_news']=daily.guardian_war_articles.gt(0).astype(int)
    # Missing coverage is not silently coded as zero observed news.
    coverage=pd.read_csv(out/'gdelt_bulk_coverage.csv')
    failed=coverage.loc[coverage.status.astype(str).ne('200'),'date']
    for date in failed:
        release=(pd.Timestamp(date)+pd.Timedelta(days=1,hours=7)).tz_localize('America/New_York')
        target=pd.Timestamp(timing_fields(release,schedule,observed)['reaction_close_session'])
        if target in daily.index:daily.loc[target,[c for c in daily if not c.startswith('guardian') and 'guardian' not in c]]=np.nan
    daily.to_csv(ROOT/'data/processed/duration_daily.csv')
    summary={'guardian_items':len(headlines),'guardian_war_items':int(headlines.war_relevant.sum()),
      'guardian_categories':headlines.category.value_counts().to_dict(),
      'gdelt_source_urls':len(source),'gdelt_categories':source.category.value_counts().to_dict(),
      'news_sessions':len(daily),'sessions_with_any_gdelt_war_news':int(daily.any_gdelt_war_news.sum()),
      'sessions_with_no_guardian_war_article':int(daily.any_guardian_war_news.eq(0).sum()),
      'guardian_clock_states':headlines.cash_equity_state.value_counts().to_dict(),
      'guardian_friday_sunday':int(headlines.is_friday_to_sunday.sum()),
      'gdelt_signal_scope':'Recent event, IRN actor, and cessation or fighting CAMEO code; geography-only records excluded from empirical counts.',
      'gdelt_strict_event_records':int((bulk.iran_actor & bulk.war_event_record).sum()),
      'gdelt_geo_only_war_records_excluded':int((~bulk.iran_actor & bulk.war_event_record).sum()),
      'interpretation':'Nominal physical-event/topic categories, not war-end probabilities or official/policy sentiment scores.',
      'caveat':'Mixed and unclear items remain explicit. A ceasefire declaration or reopened waterway does not establish permanent war termination.'}
    (out/'duration_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return headlines,source,daily,summary


if __name__=='__main__':
    *_,summary=build();print(json.dumps(summary,indent=2))
