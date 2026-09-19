import numpy as np
import pandas as pd
from src.duration_news import headline_category,canonical_url
from src.gdelt_bulk import classify_records
from src.iran_study import nearest_controls


def test_ceasefire_extension_is_not_war_extension():
    assert headline_category('Iran ceasefire extended',True)=='Cessation/reopening language'
    assert headline_category('Iran ceasefire collapses after missile attack',True)=='Continuing fighting/disruption language'


def test_unknown_and_attributed_statements_are_not_forced_into_end_probabilities():
    assert headline_category('Iran war: what happens next?',True)=='War context; duration unclear'
    assert headline_category('Trump says Iran war will end',True)=='Attributed statement/policy context'
    assert headline_category('Crude news without Iran scope',False)=='Outside Iran-war scope'


def test_meaningful_url_query_ids_survive_deduplication():
    assert canonical_url('https://news.test/?p=1&utm_source=x')!=canonical_url('https://news.test/?p=2&utm_source=x')
    assert canonical_url('https://news.test/a?utm_source=x#top')==canonical_url('https://news.test/a')


def test_archive_release_clock_is_after_news_date_and_old_mentions_do_not_enter_signal():
    data=pd.DataFrame({'event_date':['20260101','20251201','20260103'],
         'date_added':['20260101']*3,'archive_date':['2026-01-01']*3,
         'event_code':['0871','190','190'],'event_root':['08','19','19']})
    x=classify_records(data)
    assert x.war_event_record.tolist()==[True,False,False]
    assert pd.Timestamp(x.conservative_available_et.iloc[0])==pd.Timestamp('2026-01-02 07:00',tz='America/New_York')


def test_paper_controls_are_nearest_non_event_sessions_not_selected_by_returns():
    dates=pd.bdate_range('2026-01-01',periods=10)
    p=pd.DataFrame({'two_year':np.arange(10,dtype=float),'sp500':np.arange(10,dtype=float)},index=dates)
    # The selector's full-panel column audit accepts the complete outcome schema.
    from src.iran_study import OUTCOMES
    for name in OUTCOMES:
        if name not in p and name!='liquidity':p[name]=1.
    events=pd.DataFrame({'market_date':[dates[2],dates[6]]})
    a,_,audit=nearest_controls(p,events)
    p.loc[:,'sp500']=p.sp500*1000
    b,_,again=nearest_controls(p,events)
    # Tuesday is one calendar day after the Monday event; Friday is three before.
    assert audit.low_date.tolist()==[dates[3],dates[5]]
    assert a.index.tolist()==b.index.tolist()
    assert a.high.sum()==(~a.high).sum()==2
