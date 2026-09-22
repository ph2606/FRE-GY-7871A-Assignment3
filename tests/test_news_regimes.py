import numpy as np
import pandas as pd
from src.news_regimes import high_flags,build_regimes


def test_quantile_ties_and_missing_coverage_are_not_high_or_zero_imputations():
    x=pd.Series([0.,1.,2.,3.,4.,4.,8.,100.])
    complete=pd.Series([True]*7+[False])
    flags,threshold=high_flags(x,.75,complete)
    assert threshold==4
    assert flags.iloc[4]==flags.iloc[5]==0
    assert flags.iloc[6]==1 and np.isnan(flags.iloc[7])


def fixture():
    sessions=pd.to_datetime(['2026-02-27','2026-03-02','2026-03-03','2026-03-04','2026-03-05'])
    rows=[]
    for i,(stamp,session,category) in enumerate([
        ('2026-02-27T22:00:00Z','2026-03-02','Continuing fighting/disruption language'),
        ('2026-02-28T17:00:00Z','2026-03-02','Continuing fighting/disruption language'),
        ('2026-03-01T17:00:00Z','2026-03-02','Cessation/reopening language'),
        ('2026-03-03T17:00:00Z','2026-03-03','Cessation/reopening language'),
        ('2026-03-05T17:00:00Z','2026-03-05','Continuing fighting/disruption language'),
        ('2026-03-05T18:00:00Z','2026-03-05','Attributed statement/policy context')]):
        rows.append(dict(url=f'https://test/{i}',title=f'Example {i}',effective_utc=stamp,
           reaction_close_session=session,war_relevant=True,category=category,
           is_friday_to_sunday=session=='2026-03-02',cash_equity_state='weekend'))
    return pd.DataFrame(rows),sessions


def test_war_window_filters_prewar_news_and_keeps_mixed_and_no_news_explicit():
    h,sessions=fixture()
    frame,calendar,summary=build_regimes(h,sessions)
    assert len(frame)==4 and summary['war_articles_mapped']==5
    assert np.isclose(frame.iloc[0].interval_days,8/3)
    assert frame.iloc[0].direction_regime=='Mixed physical language'
    assert frame.iloc[1].direction_regime=='Good: cessation language'
    assert frame.iloc[2].direction_regime=='No retrieved war news'
    assert frame.iloc[3].direction_regime=='Bad: continuation language'
    assert calendar.market_date.tolist()==['2026-03-05']
    assert frame.iloc[3].qualitative_context_articles==1


def test_failed_page_coverage_excludes_two_possible_clock_windows():
    h,sessions=fixture()
    documents=pd.DataFrame({'url_date':['2026-03-03'],'exclusion':['download/parse failure: test']})
    frame,_,summary=build_regimes(h,sessions,documents)
    assert frame.loc['2026-03-03':'2026-03-04','high'].isna().all()
    assert summary['missing_coverage_sessions']==2
    assert (frame.loc['2026-03-03':'2026-03-04','direction_regime']=='Coverage incomplete').all()


def test_repeated_source_does_not_inflate_news_intensity():
    h,sessions=fixture()
    a,_,_=build_regimes(h,sessions)
    b,_,_=build_regimes(pd.concat([h,h.iloc[[2]]],ignore_index=True),sessions)
    pd.testing.assert_series_equal(a.war_articles,b.war_articles)
    pd.testing.assert_series_equal(a.high,b.high)
