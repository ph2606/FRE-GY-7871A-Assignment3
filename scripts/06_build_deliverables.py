"""Generate the PDF source and readable notebook from verified output tables."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.figures import build as figures
from build_notebook import build as notebook
OUT=ROOT/'outputs'
SHORT={'wti_spot':'WTI spot','brent_spot':'Brent spot','brent_wti_spread':'Brent minus WTI ($)',
 'sp500':'S&P 500','nasdaq':'Nasdaq','efa':'EFA','eem':'EEM','vix':'VIX','gold_gld':'GLD',
 'dollar':'DXY','broad_dollar':'Broad dollar','two_year':'2-year (bp)','ten_year':'10-year (bp)',
 'breakeven_10y':'Breakeven (bp)','bbb_spread':'BBB OAS (bp)','hy_spread':'HY OAS (bp)',
 'wti_futures':'WTI futures','brent_futures':'Brent futures'}


def esc(value):
    chars={'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}'}
    return ''.join(chars.get(c,c) for c in str(value))


def number(value,decimals=2):
    return '—' if pd.isna(value) else f'{value:.{decimals}f}'


def table(headers,rows,layout=None,size='small'):
    layout=layout or 'l'+'r'*(len(headers)-1)
    lines=['\\begin{center}\\'+size,'\\begin{tabular}{'+layout+'}',r'\toprule',
           ' & '.join(esc(v) for v in headers)+r' \\',r'\midrule']
    lines += [' & '.join(esc(v) for v in row)+r' \\' for row in rows]
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}\end{center}'])


SOCIAL_EXAMPLES=[
 ('2026-01-02','115824439366264186','Conditional intervention tied to treatment of protesters. Future tense and the stated condition are not evidence that intervention occurred.'),
 ('2026-01-12','115884319075881590','Announced tariff on countries trading with Iran. Trade restrictions are a distinct channel from a physical interruption of crude supply.'),
 ('2026-02-04','116013105630663812','Iran appears among several subjects in a conversation with China\u2019s president. Favorable language describes the conversation and relationship, not a measured fall in war risk.'),
 ('2026-03-01','116152251973821428','Conditional response to an anticipated Iranian action. Capitals and the conditional clause matter; keep the claimed future action separate from an observed event.'),
 ('2026-04-01','116329512466946656','A claimed ceasefire request, a Hormuz condition and continuing-attack language share one post. A single polarity label would combine distinct propositions.'),
 ('2026-05-03','116512555123589170','A shipping-operation announcement alongside negotiation references. The stated subjects include navigation access and conditions for shipping.'),
 ('2026-08-01','117023461141824050','A claimed cancellation of a planned attack is linked to an agreement condition. This is not an unconditional statement that the conflict has ended.'),
 ('2026-09-01','117196950497702512','The account reports strikes around Hormuz and states a condition for a further response. Separate the reported claim from the prospective condition.'),
 ('2026-09-07','117232304514057261','A forecast about oil and gasoline prices depends on a future war outcome. It is an attributed price claim, not an observed price effect.'),
]


def report():
    d=json.loads((OUT/'diagnostics.json').read_text())
    for path,digest in d['input_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    corpus=json.loads((ROOT/'data/raw/news/manifest.json').read_text())
    text=json.loads((OUT/'text_diagnostics.json').read_text())
    exclusions=corpus['exclusions']
    download_failures=sum(count for reason,count in exclusions.items()
                          if reason.startswith('download/parse failure:'))
    values={}
    values['CORPUS_TABLE']=table(['Step','Count'],[
      ['Unique dated URLs discovered, January–September 16',corpus['discovered_unique_2026_urls']],
      ['Excluded: live blogs or multimedia',exclusions.get('live blog or multimedia',0)],
      ['Excluded: non-news sections',exclusions.get('non-news section',0)],
      ['Excluded: failed downloads or parsing',download_failures],
      ['Eligible extracted news documents',text['eligible_news_documents']],
      ['Excluded: latest body revision after cutoff',text['modified_after_cutoff']],
      ['Documents within the information cutoff',text['within_cutoff_documents']],
      ['Iran/conflict-relevant documents',text['war_relevant_documents']],
      ['Relevant documents with at least 20 eligible tokens',text['numeric_text_eligible']]])
    m=pd.read_csv(OUT/'news_monthly.csv')
    values['NEWS_MONTHLY']=table(['2026','Articles','War-related','Sessions','Attention','Negative %','Uncertain %'],
       [[r.date[5:],r.articles,r.war_articles,r.sessions,number(r.attention),number(r.negative_pct),number(r.uncertainty_pct)] for r in m.itertuples()],size='footnotesize')
    h=pd.read_csv(OUT/'heteroskedasticity.csv')
    values['HET_TABLE']=table(['Outcome','Pairs','d1','d2','Pooled','Pooled 95% interval','W'],
       [[SHORT[r.y],r.n_pairs,number(r.d_reference_scenario),number(r.d_outcome_scenario),number(r.d_pooled_scenario),
         '—' if r.weak_identification else '['+number(r.d_pooled_scenario_ci_low)+', '+number(r.d_pooled_scenario_ci_high)+']',
         'W' if r.weak_identification else ''] for r in h.itertuples()],layout='lrrrrll',size='footnotesize')
    robust=pd.read_csv(OUT/'heteroskedasticity_robustness.csv')
    rows=[['Centered baseline',24]+[number(h.set_index('y').at[y,'d_pooled_scenario']) for y in ['wti_spot','brent_spot','sp500','ten_year']]]
    names={'uncentered_second_moments':'Original second moments','exclude_fomc_pairs':'Exclude FOMC pairs',
           'shift_uncertain_events':'Shift uncertain events','within_month_news_regimes':'Within-month NLP regimes',
           'january_may':'January–May','june_september':'June–September'}
    for key,label in names.items():
        a=robust.loc[robust.specification.eq(key)].set_index('y')
        rows.append([label,int(a.at['wti_spot','n_pairs'])]+[number(a.at[y,'d_pooled_scenario']) for y in ['wti_spot','brent_spot','sp500','ten_year']])
    values['ROBUST_TABLE']=table(['Specification','WTI pairs','WTI','Brent','S&P','10y bp'],rows,size='footnotesize')
    r=pd.read_csv(OUT/'news_regressions.csv')
    daily=pd.read_csv(ROOT/'data/processed/news_daily.csv')
    corr=pd.read_csv(OUT/'benchmark_correlations.csv',index_col=0)
    trend=pd.read_csv(OUT/'trend_tests.csv').set_index(['feature','term'])
    values.update({'PRIMARY_TESTS':str(len(r)), 'PRIMARY_OUTCOMES':str(r.outcome.nunique()),
      'REG_MIN_N':str(r.n.min()), 'REG_MAX_N':str(r.n.max()),
      'NUMERIC_DOCS':str(text['numeric_text_eligible']), 'LEXICAL_SESSIONS':str(daily.negative_pct.notna().sum()),
      'NEWS_SESSIONS':str(len(daily)), 'BENCH_N':str(d['benchmark_n']),
      'BENCH_ATT':number(corr.at['GPRD','log_attention'],3), 'BENCH_NEG':number(corr.at['GPRD','negative_pct'],3)})
    for name,feature,term in [('ATT_STEP','log_attention','active_conflict'),
          ('ATT_TREND','log_attention','time_months'),('NEG_TREND','negative_pct','time_months'),
          ('NEG_STEP','negative_pct','active_conflict'),('UNC_TREND','uncertainty_pct','time_months'),
          ('UNC_STEP','uncertainty_pct','active_conflict')]:
        row=trend.loc[(feature,term)]
        values[name]=number(row.coefficient,3)
        values[name+'_P']=number(row.p_value,3)
        values[name+'_Q']=number(row.q_value,3)
    rows=[]
    for y in r.outcome.unique():
        a=r.loc[r.outcome.eq(y)].set_index('feature');aa=a.loc['d_log_attention'];nn=a.loc['d_negative_pct']
        rows.append([SHORT[y],int(aa.n),f'{aa.coefficient:+.3f} ({aa.p_value:.3f})',f'{nn.coefficient:+.3f} ({nn.p_value:.3f})',f'{aa.q_value:.3f} / {nn.q_value:.3f}'])
    values['REG_TABLE']=table(['Outcome','N','Attention','Negative vocabulary','qA / qN'],rows,layout='lrlll',size='footnotesize')
    values['P_UNADJUSTED']=str(d['primary_news_unadjusted_p_lt_05'])
    values['MIN_Q']=f'{r.q_value.min():.3f}'
    values['INVEST_TABLE']=table(['Assumed yield shift','Bill price effect','Portfolio effect'],
      [['−100 bp','+0.250%','+0.025 pp'],['0 bp','0.000%','0.000 pp'],['+100 bp','−0.250%','−0.025 pp']],layout='lrr')
    examples=[r'\begin{longtable}{p{.14\linewidth}p{.79\linewidth}}',r'\toprule Date (ET) & Reading of the statement \\ \midrule\endhead']
    for date,post,reading in SOCIAL_EXAMPLES:
        examples.append(esc(date)+' & '+r'\href{https://truthsocial.com/@realDonaldTrump/'+post+r'}{Original post.} '+esc(reading)+r' \\[4pt]')
    values['SOCIAL_EXAMPLES']='\n'.join(examples+[r'\bottomrule\end{longtable}'])
    events=pd.read_csv(OUT/'event_calendar.csv');match=pd.read_csv(OUT/'matching_audit.csv').set_index('high_date')
    lines=[r'\small\begin{longtable}{p{.115\linewidth}p{.115\linewidth}p{.115\linewidth}p{.55\linewidth}}',
           r'\toprule Event date & US session & Comparison & Development and source \\ \midrule\endhead']
    for e in events.itertuples():
        date=e.market_date[:10];low=match.at[date,'low_date'] if date in match.index else 'Excluded'
        uncertain=str(e.timing_uncertain).lower()=='true'
        lines.append(esc(e.event_date)+' & '+esc(date+(' ?' if uncertain else ''))+' & '+esc(str(low)[:10])+' & '+
           r'\href{'+e.source_url+r'}{'+esc(e.description)+r'} \\[3pt]')
    values['EVENT_TABLE']='\n'.join(lines+[r'\bottomrule\end{longtable}\normalsize'])
    template=(ROOT/'scripts/report_template.tex').read_text(encoding='utf-8')
    for key,value in values.items():template=template.replace('%%'+key+'%%',value)
    assert '%%' not in template,'Unresolved report placeholder'
    (ROOT/'report.tex').write_text(template,encoding='utf-8')
    return d,values


if __name__=='__main__':
    figures()
    diagnostics,values=report()
    notebook(ROOT,diagnostics,SOCIAL_EXAMPLES,values)
    print('Built report.tex and assignment3.ipynb from verified results.')
