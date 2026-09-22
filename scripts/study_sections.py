"""Standalone narrative shared by the PDF and executable notebook."""
import json
import pandas as pd


def sections(root):
    from study_content import SOCIAL_EXAMPLES,READINGS
    out=root/'outputs'
    d=json.loads((out/'study_diagnostics.json').read_text())
    n=d['nlp_regimes']
    vals={'PAIRS':str(d['main_pairs']),'HIGH':str(n['high_sessions']),'LOW':str(n['low_sessions']),
          'THRESHOLD':f"{n['threshold_articles_per_calendar_day']:.2f}",'ARTICLES':str(n['war_articles_mapped']),
          'WEAK':str(d['main_weak_rows']),'RANK':f"{d['rank_one_residual']:.3f}",
          'LPQ':f"{d['lp_min_primary_q']:.3f}",'PRIOR':str(d['lp_placebo_q_lt05'])}
    s=[]
    def add(title,text,code='',display=None,page=True):
        for k,v in vals.items():text=text.replace('@@'+k+'@@',v)
        assert '@@' not in text
        s.append(dict(title=title,text=text.strip(),code=code.strip(),display=display,page=page))
    add('Research question and principal finding',r'''
How does unusually intensive Iran-war reporting change the covariance structure of global financial markets? The main sample starts with the outbreak on **28 February 2026** and ends on **18 September**, the latest completed US cash-market session when inputs were frozen before September 21 trading. February 27 closes supply the prewar baseline; March 2 supplies the first post-outbreak return. January–February information provides context and the broad social appendix, not observations in the primary financial estimates.

The study first implements the uncentered second-moment and instrumental-variable method in Rigobon and Sack’s *The Effects of War Risk on U.S. Financial Markets*. **NLP determines high-news (1) and low-news (0) days; market data determine responses. Story direction is unnecessary for this identification.** Tables 1–3 retain the paper’s purposes: a high-news chronology, three IV sensitivity estimates and conditional variance calculations.

Relevant publisher coverage above its active-window 75th percentile identifies **@@HIGH@@ high-news and @@LOW@@ low-news sessions**. Fitted Treasury availability and interval alignment leave **@@PAIRS@@ H/L pairs**. Anchor variance rises, but a positive shift alone does not establish an isolated war shock. The rank-one restriction fits poorly, some identifying moments disagree and results depend on source, threshold and timing.

From the prewar close, WTI spot rises **59.83%** and Brent **83.40%** through September 15, while the spot spread widens from **$4.36 to $23.78 per barrel**. Ten-year CMT yields rise 97 basis points through September 17. These are observed trends, not causal war effects. The report distinguishes the Iraq benchmark, Iran hypotheses and actual estimates—including failures to match expected signs. Directional news, novel language and the 404-record Truth Social corpus are separate extensions.
''',code='''from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from IPython.display import display,Image
ROOT=Path.cwd()
assert (ROOT/'src').is_dir(), 'Execute from the repository root.'
sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs'
pd.set_option('display.max_rows',60)
pd.set_option('display.max_columns',18)
pd.options.display.float_format='{:,.4f}'.format
from src.text_features import build as build_text
from src.social_audit import main as audit_social
from src.social_timing import build as build_social_timing
from src.duration_news import build as build_duration
from src.news_regimes import build as build_news_regimes
from src.phrase_audit import build as build_phrase_audit
from src.iran_study import run
from src.study_figures import build as build_figures
build_text()
audit_social()
social_audit,social_calendar,social_sessions,social_summary=build_social_timing()
headlines,gdelt_sources,duration_daily,duration_summary=build_duration()
regimes,high_news_calendar,regime_summary=build_news_regimes()
phrase_documents,phrase_terms,phrase_examples,phrase_summary=build_phrase_audit()
diagnostics=run(bootstrap=1999)
build_figures()
for name,digest in diagnostics['input_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
t1=pd.read_csv(OUT/'paper_table1.csv')
t2=pd.read_csv(OUT/'paper_table2.csv')
t3=pd.read_csv(OUT/'paper_table3.csv')
print('Analysis recomputed from cached inputs; hashes verified.')
''',page=False)
    add('Data, observation windows and free news access',r'''
The references are [Rigobon and Sack (2003), NBER Working Paper 9609](https://www.nber.org/papers/w9609) and [Rigobon (2003), Review of Economics and Statistics 85(4), 777–792](https://doi.org/10.1162/003465303772815727). The war paper selected 17 news days within 47 business days before Iraq’s invasion, including some other security developments. Here Tables 1–3 use the active Iran-conflict window.

All 18 market feeds were refreshed. Exchange-traded series and breakeven inflation reach September 18; CMT yields and credit OAS September 17; spot oil September 15; the broad dollar and fitted off-the-run curve September 11. Nothing is forward-filled. Prices use simple percentage returns, $r_t=100(P_t/P_{t-1}-1)$; yields/OAS use percentage-point changes; dollar oil and the physical spread use dollars per barrel. Each one-session outcome must share current and preceding dates with the reference; incomplete H/L pairs are removed together.

The main Treasury inputs are Gürkaynak–Sack–Wright coupon-equivalent **par yields**, SVENPY02 and SVENPY10. The Fed model excludes on-the-run and first-off-the-run notes and bonds, giving a closer conceptual match to the paper than CMT yields, while remaining a revisable research estimate. [Fed curve documentation](https://www.federalreserve.gov/data/nominal-yield-curve.htm). The original 12-month oil contract, dollar gold and traded liquidity premium are not identically reproduced. Nearby WTI, GLD returns and OAS are disclosed proxies; unavailable premium cells stay blank.

**GDELT is the free historical connector for the broad-source comparison.** All 261 daily bulk files for January 1–September 18 were obtained. The broad Iran actor/location extract has 1,814,920 event rows and 372,908 canonical URLs. These are NLP event metadata and links, not downloaded full articles. Empirical counts further require an Iranian actor, a recent event and eligible CAMEO codes; 57,410 geography-only war-coded rows are excluded. An Iranian actor still does not prove this war is the subject. [GDELT access](https://www.gdeltproject.org/data.html); [event codebook](https://data.gdeltproject.org/documentation/GDELT-Data_Format_Codebook.pdf).

The **primary text regime** uses publisher text for explicit relevance and better timestamps. The Guardian archive has 1,896 URLs: 444 live/multimedia and 417 non-news exclusions leave 1,035 documents, with no unresolved download failures. NLP identifies 926 relevant January-context articles; **@@ARTICLES@@ map to active-conflict return windows**. Relevance requires Iran and conflict/energy context in a headline or at least two body sentences. This is one English-language outlet and a heuristic classifier, not a global census or independently validated label set.

GDELT’s keyless DOC API supports discovery but hit caps/rate limits; bulk files were practical. The Guardian academic/noncommercial API needs a key and permits 500 requests/day; pages here were collected directly. NewsAPI’s free developer tier has 100 requests/day, a 24-hour delay and one month of history. [Guardian terms](https://open-platform.theguardian.com/access/); [NewsAPI plans](https://newsapi.org/pricing).
''',code='''market_manifest=json.loads((ROOT/'data/raw/market/manifest.json').read_text())
display(pd.DataFrame(market_manifest['series'])[['series','provider','first','last','observations_2026']])
display(pd.read_csv(ROOT/'data/processed/offrun_yields.csv',index_col=0).tail())
print(json.dumps(duration_summary,indent=2))
print(json.dumps(regime_summary,indent=2))
coverage=pd.read_csv(OUT/'gdelt_bulk_coverage.csv')
assert len(coverage)==261 and coverage.status.astype(str).eq('200').all()
''',display='data')
    add('Oil prices, the spread and the yield curve',r'''
February 27 spot prices are $66.96 for WTI and $71.32 for Brent; their September 15 observations are $107.02 and $130.80. Define $S_t=P_t^{Brent}-P_t^{WTI}$ and $p_t=100(P_t^{Brent}/P_t^{WTI}-1)$. The spread rises by **$19.42 per barrel**, and the relative premium from **6.51% to 22.22% of WTI**. Geography, quality, inventories, transport and refining also affect it; this is not a pure war premium or probability.

Nearby rolling futures end September 18 at $100.30 and $103.87, gains of 49.66% and 43.31%. Their $3.57 difference is not the September 15 spot spread and may compare different delivery months. Nearby oil is also more exposed to immediate physical constraints than the paper’s year-ahead contract. A spread trade requires matched expiries and basis analysis. [EIA on physical prices and contract timing](https://www.eia.gov/todayinenergy/detail.php?id=67424).

The two-year CMT rises from 3.38% to 4.67%, and the ten-year from 3.97% to 4.94%, through September 17. The 10y-minus-2y curve falls from 59 to 27 basis points: **32 bp of bear flattening**. Yield changes are not bond returns. From the same baseline, S&P gains 11.22%, Nasdaq 17.00%, EFA 1.19% and EEM 7.67%; GLD falls 17.07%. ETFs are adjusted fund-price proxies; equity indexes are price indexes. Full-period movements include other shocks and differ from isolated war-news effects.
''',code='''display(Image(filename=str(OUT/'study_oil.png')))
display(Image(filename=str(OUT/'study_markets.png')))
from src.iran_study import market_panel
_,market_values,market_levels=market_panel('cmt')
rows=[]
for name in ['wti_spot','brent_spot','wti_futures','brent_futures','sp500','nasdaq','efa','eem','gold_gld']:
    x=market_values[name].loc['2026-02-27':'2026-09-18'].dropna()
    rows.append([name,str(x.index[0].date()),str(x.index[-1].date()),x.iloc[0],x.iloc[-1],100*(x.iloc[-1]/x.iloc[0]-1)])
display(pd.DataFrame(rows,columns=['Market','Baseline','Endpoint','Baseline level','Last level','Simple change %']))
''',display='trends')
    add('NLP high-news days and original-paper identification',r'''
Let $N_t$ be distinct relevant publisher articles in a close-to-close information window and $D_t$ its elapsed hours divided by 24. The effective clock is the later of publication and modification, which handles known edits conservatively without reconstructing original real-time bodies. The first news window starts at February 28 midnight ET; its return uses Friday’s close. Set $A_t=N_t/D_t$ and $H_t=1$ when $A_t>Q_{0.75}(A)$, otherwise 0. The threshold is **@@THRESHOLD@@ articles per elapsed day**; ties stay low. This retrospective rule is not a forecast or preregistered threshold and is not tuned against returns.

There are **@@HIGH@@ high and @@LOW@@ low sessions** among 140 active-conflict sessions. Match each estimable H without replacement to the nearest eligible L by calendar distance, preferring the earlier date on a tie. Fitted-curve data supply **@@PAIRS@@ pairs**. The longest gap is 38 days, weakening the unchanged-nuisance-volatility assumption; within-month thresholds test sensitivity. Low means lower retrieved coverage, not fundamentals-only trading. Only four sessions have no relevant publisher article. Failed coverage would be missing rather than zero.

Write $\Delta X_t=d w_t+u_t$, normalizing the two-year loading to one. With one changing orthogonal war-shock variance, stable loadings and unchanged nuisance moments,

$$\Omega_H-\Omega_L=\delta dd',\qquad\delta>0.$$

Identification comes from this covariance shift and **does not require story direction**. Robust OLS errors do not provide identification. Following the paper, use zero-mean, uncentered moments and no intercept. For two-year change $x$ and outcome $y$, define $a=E_H[x^2]-E_L[x^2]$, $b=E_H[xy]-E_L[xy]$ and $c=E_H[y^2]-E_L[y^2]$. Then $\widehat d_1=b/a$ and $\widehat d_2=c/b$. Instruments are $s_tx_t$ and $s_ty_t$, with $s_t=+1$ on H and −1 on L; pooled 2SLS uses both.

**Table 1 is the NLP high-news chronology.** It shows session, article count/intensity, comparison date and the earliest effective-clock relevant headline as an illustrative source. That headline alone does not define H. Full article/session audits remain local. The manually sourced event calendar is a separate robustness and weekend-timing check.
''',code='''display(t1[['market_date','war_articles','intensity','friday_sunday_articles','description','source_url','included_in_main_estimation']])
display(pd.read_csv(OUT/'paper_matching.csv'))
display(pd.read_csv(OUT/'nlp_regimes_monthly.csv'))
display(regimes[['war_articles','interval_days','intensity']].describe())
display(Image(filename=str(OUT/'study_news.png')))
''',display='table1')
    add('Table 2: three IV sensitivity estimates',r'''
Each effect is $-0.25\widehat d$, the paper’s **25-bp two-year-yield decline**. Inputs are percentage-point changes, so the multiplier is −0.25. This fixes units without establishing which factor orientation means greater Iran-war risk.

Parentheses show absolute HC1 IV t-statistics with no-intercept finite-sample correction. The paper does not specify its covariance estimator sufficiently to reproduce that convention exactly; classical t-statistics remain in the notebook. Bootstrap diagnostics use 1,999 independent within-H/L resamples, fixed seeds and common draws for identical anchor samples. They are supplementary, not a source-paper recipe or weak-IV-robust confidence sets.

**@@WEAK@@ of 19 available rows** have a reference-variance interval crossing zero and are marked W; unreliable ratio intervals are withheld. Other rows still require valid instruments and stable restrictions. Pooled responses are **−0.162 pp for the ten-year yield, +1.99% for S&P, +13.67% for WTI spot and −2.94% for Brent spot**. Opposing oil signs and disagreement among moments challenge a common one-factor interpretation. Conventional significance cannot repair invalid exclusions or unstable transmission.
''',code='''display(t2[['label','unit','n_high','omega1_effect','omega1_abs_t_hc1','omega2_effect','omega2_abs_t_hc1','pooled_effect','pooled_abs_t_hc1','weak_reference_variance_shift']])
display(t2[['label','pooled_abs_t_classical','pooled_abs_t_hc1','pooled_effect_ci_low','pooled_effect_ci_high','availability_reason']])
display(pd.read_csv(OUT/'paper_diagnostics.csv'))
''',display='table2')
    add('The Iraq benchmark and Iran expectations',r'''
The original pooled response to the −25-bp two-year scenario was ten-year yields **−26 bp**, breakevens **−11 bp**, 12-month oil **+$0.77/barrel**, BBB spreads **+5 bp**, high-yield spreads **+34 bp**, S&P **−3.76%**, and a statistically small gold-dollar response. These are historical estimates, not coefficients imposed on Iran.

For adverse Iranian supply disruption, plausible hypotheses are oil up, wider credit spreads and lower equities. **Treasury yields may rise** if inflation and financing pressures outweigh safe-asset demand; breakevens may also rise. Debt and weaker flight to quality are hypotheses, not isolated mechanisms in these regressions. Gold, currencies and the relative oil spread need not have universal signs.

The comparison shows a **+25-bp two-year scenario** for Iran beside the original results. It negates the same estimates; standard-error magnitudes and absolute t-statistics are unchanged, and variance shares are unchanged because they use squared loadings. The ten-year becomes +0.162 pp and S&P −1.99%, agreeing in sign with those hypotheses. WTI spot instead falls 13.67%, Brent rises 2.94%, and the high-yield spread narrows. These counterexamples remain visible: changing orientation cannot make every outcome agree.

Sign agreement is not statistical confirmation. Original gold is dollars per ounce versus GLD percent; original oil is year-ahead versus nearby dollars. Units and contracts are explicit. Covariance shifts alone do not identify the semantic sign of a latent shock or establish that it represents only war risk.
''',code='''display(pd.read_csv(OUT/'benchmark_expectations.csv'))
rise=pd.read_csv(OUT/'iran_yield_rise_scenario.csv')
display(rise[['label','unit','pooled_effect','pooled_abs_t_hc1','weak_reference_variance_shift']])
for method in ['omega1','omega2','pooled']:
    assert np.allclose(rise[method+'_effect'],-t2[method+'_effect'],equal_nan=True)
    assert np.allclose(rise[method+'_abs_t_hc1'],t2[method+'_abs_t_hc1'],equal_nan=True)
''',display='benchmark')
    add('Table 3: conditional variance calculations',r'''
The incremental contribution is $q_j=\widehat d_{pool,j}^{\,2}a$. Its H share is $100q_j/E_H[y_j^2]$; the full aligned war-window share is $100n_Hq_j/\sum_{t\in all}y_{jt}^2$. The denominator uses **all aligned days**, not just selected controls. Moments/contributions have squared outcome units; shares are percentages. Attribution cells for the reference yield and unavailable premium remain blank.

Stable nuisance moments, one changing orthogonal war factor and serial independence give the incremental variance a conditional lower-bound interpretation. Positive a does not establish those restrictions. These are **conditional diagnostics, not proven causal shares**; negative or above-100% estimates would remain visible.

The two-year H mean square is about 1.97 times L. Its increase is 0.001519 pp², with a bootstrap interval near [0.000082, 0.003186] pp² for the complete 34-pair sample: positive but close to zero at the lower endpoint. Ten-year H/full-window shares are **21.18%/6.67%**, versus 84.7%/62.8% in 2003. S&P shares are **10.54%/3.23%**, versus 34.2%/19.7%. Its H mean square is below L despite a positive model-implied contribution, cautioning against literal one-factor attribution.
''',code="display(t3[['label','unit_squared','n_high','variance_low','variance_high','predicted_variance_change','high_variance_share_pct','all_variance_share_pct','full_period_n','full_period_n_high']])",display='table3')
    add('Identification diagnostics and robustness',r'''
The six-market standardized second-moment contrast has multiple positive and negative eigenvalues. Its distance from the nearest positive rank-one matrix is **@@RANK@@ of its Frobenius norm**, a diagnostic rather than a formal rank-test p-value. Positive anchor variance and a significant ten-year t-statistic do not validate the whole system.

Checks use CMT yields; the curated chronology in the same war window; two-thirds/80th-percentile NLP thresholds; within-month thresholds; next-session timing; controls excluding adjacent high days; stricter GDELT Iran-actor counts; and pairs excluding scheduled FOMC dates. No return magnitudes select labels. GDELT and publisher intensity share 26 high sessions but differ in coverage and clocks.

The ten-year response is comparatively stable under several threshold choices, while equity/oil estimates vary materially. Next-session alignment produces large unstable oil ratios; within-month classification changes the S&P sign. Gaps up to 38 days cannot ensure unchanged non-war volatility. These findings motivate alternative designs rather than relying on the strongest individual t-statistic.
''',code='''rank=json.loads((OUT/'paper_rank.json').read_text())
print('Eigenvalues:',np.round(rank['eigenvalues'],4),'Relative residual:',rank['rank_one_residual'])
display(Image(filename=str(OUT/'study_identification.png')))
robust=pd.read_csv(OUT/'paper_robustness.csv')
display(robust.loc[robust.variable.isin(['ten_year','sp500','wti_spot','brent_spot']),['specification','label','n_high','pooled_effect','weak_reference_variance_shift']])
''',display='robust')
    add('Why the 2026 economy can produce a different table',r'''
The 2003 sample followed the dot-com collapse, September 11, accounting scandals and weak capital spending. It was a fragile recovery after the March–November 2001 recession. The [Federal Reserve’s February 2003 report](https://www.federalreserve.gov/boarddocs/hh/2003/february/ReportSection1.htm) records a 1.25% funds target and economic slack; [NBER’s chronology](https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions) dates the recession. Demand weakness and safe-asset demand can explain falling yields, but the paper does not separately identify those channels.

Annual US crude production rises from **5.649 million barrels/day in 2003 to 13.662 in 2025**, or 141.8%; June 2026 is 13.792. Net imports of crude and petroleum products change from 11.238 million barrels/day in 2003 to net exports of 2.848 in 2025. Relative to products supplied, these are +56.1% and −13.7%. Crude-only net imports remain positive at 1.309 million barrels/day in June 2026. These are different trade concepts. [EIA production](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=MCRFPUS2&f=A), [monthly production](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=MCRFPUS2&f=M), [total net imports](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=MTTNTUS2&f=A), [products supplied](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=MTTUPUS2&f=A), [crude net imports](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=MCRNTUS2&f=M).

A larger producer sector can offset some importing-firm and consumer losses, but US consumers still face global oil prices. Pre-conflict Hormuz flows were 20.9 million barrels/day in first-half **2025**, about 20% of global petroleum-liquids consumption; these are exposure figures, not 2026 traffic estimates. [EIA prices](https://www.eia.gov/energyexplained/oil-and-petroleum-products/prices-and-outlook.php); [chokepoints](https://www.eia.gov/international/content/analysis/special_topics/World_Oil_Transit_Chokepoints/).

January 2003 headline/core CPI inflation was 2.6%/1.9% year over year; August 2026 was 3.4%/2.4%, with energy up 16.3% and gasoline 27.4%. These observations motivate an inflation channel without attributing all inflation to war. [BLS 2003](https://www.bls.gov/news.release/archives/cpi_02212003.pdf); [BLS 2026](https://www.bls.gov/news.release/archives/cpi_09112026.htm). The BIS discusses energy-driven rate repricing and later real-term-premium pressures associated with fiscal concerns and issuance. Such channels can weaken the net flight-to-quality response, but these estimates do not isolate them. [BIS annual report](https://www.bis.org/publications/aer-2026/progress-peril); [September review](https://www.bis.org/publications/qr-202609/yields-climb-yet-risk-appetite-holds-firm).

On March 14, 2003, public CMT yields were 1.56% at two years and 3.72% at ten years, a 216-bp slope; September 17, 2026 has a 27-bp slope. [Historical H.15](https://www.federalreserve.gov/releases/h15/20030317/). The 2026 bear flattening can reflect expected short rates; ten-year yields also include inflation compensation and term/liquidity premia. The September 16 FOMC announcement at 14:00 ET raised the funds target 25 bp to 3.75–4.00%, effective September 17. CMT yields rose 7 bp at two years and 1 bp at ten years on announcement day, then fell 7 bp each the next day. Daily changes cannot isolate the announcement’s effect. [FOMC statement](https://www.federalreserve.gov/newsevents/pressreleases/monetary20260916a.htm); [implementation](https://www.federalreserve.gov/newsevents/pressreleases/monetary20260916a1.htm).

Differences from Iraq can therefore reflect supply exposure, inflation and financing conditions, expected war duration, news composition, instrument maturity and identification failure. Economic explanations do not validate an unstable coefficient automatically.
''')
    add('Good, bad and no-news states',r'''
The directional extension classifies **physical-event language** without observing returns. Continuing fighting/disruption is the “bad” category; cessation/reopening is “good.” These names concern conflict persistence, not political merit. A source reporting both belongs to mixed; attributed claims, negotiations without cessation and unclear reports remain separate. A declaration does not prove hostilities have stopped. Every eligible headline receives a category, including out-of-scope material.

The active window contains **39 continuation, 5 cessation, 4 no-retrieved-war-news, 12 mixed and 80 unclear/attributed sessions**. There is no universal zero-news state in the broad GDELT feed. The four publisher zeros are outlet-level absences and cannot establish fundamentals-only trading. Low intensity in the primary model includes ordinary war coverage and is different from no news.

The table reports conditional daily means, with actual observation counts. Continuation days average higher oil and ten-year yields and lower equities, as hypothesized. That association neither validates the text labels nor establishes causation: macroeconomic releases, reporting selection and prior market movements can coexist. Five cessation and four zero-news sessions provide too little support for a credible unrestricted three-state covariance model. Mixed and unclear days must not be forced into favorable or adverse classes.

A three-state IH extension would compare each active-news covariance with a shared low-news covariance, test common loadings, and resample shared controls jointly. Centered covariance would explicitly differ from the paper’s raw second moments, since directional states can have different means. More regimes help only with adequate counts and independent variance shifts. Rigobon’s order condition for three observed variables and one unrestricted common shock requires at least four regimes, plus a rank condition; three labels alone do not identify that system.
''',code='''display(regimes.direction_regime.value_counts().rename('Sessions').to_frame())
direction=pd.read_csv(OUT/'directional_regime_markets.csv')
display(direction[['regime','sessions','outcome','observations','mean_change','median_change','sd_change','unit']])
assert regimes.high.eq(1).sum()==35 and regimes.high.eq(0).sum()==105
assert not regime_summary['selection_uses_market_returns']
''',display='direction')
    add('New war language and classification errors',r'''
A dictionary written for Iraq can miss Iranian facilities, Strait of Hormuz access, tanker disruption, drones and operation names. The audit compares a simple headline rule requiring Iran alongside war, conflict, fighting or military with an expanded rule covering relevant entities, physical action, shipping, energy, nuclear facilities, diplomacy and exact operation names. The comparator is an audit baseline, not a dictionary claimed to come from Rigobon and Sack or the primary body-text classifier.

Among 1,035 source documents, the comparator retrieves 244 and the expansion 642: **398 additional candidates**, with no comparator matches lost. More retrieval is not measured accuracy. Some additions concern background energy or nuclear issues; exact operation names can refer to earlier events. “New” here means newly retrieved relative to this comparator, not proof of newly coined wording.

The audit distinguishes extending a ceasefire, extending a deadline and extending a war. It flags negated ending/collapse language, conditional or modal wording, quotations, labor context and ambiguous grammatical targets. URLs are provenance, not classification text. There are 105 conditional/modal and 300 quotation cues; counts overlap and do not represent independent shocks.

Actual failures remain instructive. A headline describing Democrats renewing a push to curb the war can be attached incorrectly to war extension; stepping back from renewed conflict is not a fresh extension; a ceasefire-extension reference can concern Israel–Hezbollah rather than Iran. Nearest-target matching can also fail with multiple nouns, clause boundaries and unrelated negation. These documented errors preclude treating cues as verified duration news. Source-linked examples and all matched spans remain in the local audit.

The 17 constructed cases test software behavior; they do not estimate empirical precision, recall or probability calibration. Modern language models also face genuinely new phrases, misleading attribution and domain shift. A stronger evaluation would label a held-out, time-stratified sample independently, include mixed and uncertain classes, report class-specific precision/recall and agreement, and freeze the rule before future testing. No such independent accuracy estimate is claimed here.
''',code='''print(json.dumps({k:phrase_summary[k] for k in ['distinct_source_documents','legacy_retrieved_documents','expanded_retrieved_documents','newly_retrieved_distinct_documents','accuracy_estimated','synthetic_software_checks']},indent=2))
display(pd.DataFrame(phrase_summary['cue_document_counts'].items(),columns=['Review cue','Documents']))
display(phrase_terms)
display(phrase_examples)
display(pd.read_csv(OUT/'phrase_audit_stress_cases.csv'))
''',display='phrases')
    add('Friday–Sunday news and market reaction windows',r'''
The reference clock is NYSE **09:30–16:00 New York time**, using the holiday calendar and daylight-saving conversion. An item maps to the first scheduled close strictly after its timestamp. Before-open and intraday news can enter the same session; news at or after the close enters the next. Friday after-close, Saturday and Sunday usually map to Monday, with holiday adjustments. A separate first-open clock and following reaction session are retained. [NYSE calendar](https://www.nyse.com/trade/hours-calendars).

The February 28 Saturday outbreak maps to March 2. The April 7 announcement at 18:32 ET maps to April 8; April 12 Sunday information maps to April 13. Known report times establish availability, not necessarily the first underlying event. Date-only curated events retain timing uncertainty. Their audit shows both the mapped session and the following session; the primary NLP model also receives a one-session timing sensitivity check.

The January-context publisher sample has **370 Friday–Sunday items**; Truth Social has **162**. The social set contains 128 market-open, 82 pre-open, 82 after-close, 101 weekend and 11 holiday records, including 15 Friday after-close posts. Publisher timing uses the later publication/modification time; social timing uses the original post clock. Neither necessarily identifies first public information, particularly for reposts.

Cash-equity closure is not closure of every market. WTI futures normally reopen Sunday at 18:00 ET; Treasury CMT quotes are indicative bids around 15:30 ET. Daily closing data cannot recover synchronized intraday responses or separate opening gaps from daytime moves. [CME WTI hours](https://www.cmegroup.com/education/courses/event-contracts-underlying-markets/wti-overview); [Treasury methodology](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics).

GDELT bulk dates lack publisher timestamps. The conservative project clock is **07:00 ET the next calendar day**, followed by the next US close. One source URL is counted once per availability session but can recur on later days. The dynamic model consequently studies archive-availability associations. It cannot treat those files as a previous-day trading signal.
''',code='''display(Image(filename=str(OUT/'study_timing.png')))
reactions=pd.read_csv(OUT/'event_reaction_windows.csv')
display(reactions.loc[reactions.friday_sunday & reactions.outcome.eq('sp500'),['event_date','cash_window','reaction_session','following_session','reaction_change','following_change','unit']])
display(reactions.loc[reactions.event_date.eq('2026-04-07'),['event_date','outcome','reaction_session','following_session','reaction_change','following_change','unit']])
display(pd.DataFrame(social_summary['equity_state_counts'].items(),columns=['Equity state','Posts']))
''',display='timing')
    add('Is heteroskedasticity identification the best approach?',r'''
**It is a useful benchmark, but this sample does not justify treating it as the best causal estimator.** Its advantage is that text needs only to identify greater war-shock variance, avoiding unreliable story-direction scores. Its demanding restrictions are stable transmission, unchanged nuisance moments, orthogonality and a sufficiently strong variance shift. Concurrent inflation, monetary-policy and energy-supply shocks make those restrictions difficult during an ongoing war. The rank and sensitivity diagnostics are material evidence against an unqualified one-factor interpretation.

The implemented alternative keeps two physical-news dimensions: cessation-coded and fighting-coded GDELT sources. Define $z_t^k=\Delta\log(1+N_t^k)/sd[\Delta\log(1+N_t^k)]$. Counts require an Iranian actor and an event dated zero to two days before archive entry. CAMEO 087 includes de-escalation/declarations; 19/20 cover fighting/violence. Both can occur in a mixed source. [CAMEO manual](https://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf). Scaling is retrospective, not a trading calibration.

For eight outcomes, local projections estimate cumulative responses from the preceding close through horizons 0, 1 and 5 sessions, beginning March 2. Controls include both news series and their lags, a lagged outcome, log calendar-interval length, month effects and FOMC dates. Price responses are cumulative simple percentages; yields and spreads use their stated difference units. HAC inference uses 5+h lags, finite-sample correction and t inference. Forty-eight primary coefficients form one Benjamini–Hochberg family; adjustment does not remove endogeneity.

**None of 48 primary coefficients has adjusted q below 0.05**; the smallest is **@@LPQ@@**. This is limited evidence, not proof of no market effect. Sixteen separate prior-return checks use a lag-two outcome control to avoid regressing a variable on itself. **@@PRIOR@@ survive adjustment**: preceding WTI and Brent increases and a preceding S&P decline associated with fighting coverage. Movements before archive availability are consistent with earlier public news, common shocks or reporting following markets. These results cannot support a causal or immediately tradable news strategy.

A more suitable causal design would verify first-public timestamps and use narrow windows in synchronized Treasury, equity and matched-maturity oil futures. Separate unexpected cessation, supply interruption and renewed fighting; retain uncertainty; screen overlapping CPI/FOMC releases; distinguish Sunday futures from Monday cash trading. This improves timing but still requires clean windows and unanticipated information. Complete historical intraday quotes and reliable first-report clocks were not acquired, so this remains a proposal. [Federal Reserve announcement-window study](https://www.federalreserve.gov/pubs/ifdp/2006/886/ifdp886.htm).

External-instrument local projections could trace dynamics if a **signed surprise** were relevant to war risk and orthogonal to other shocks. The unsigned high-news indicator can change variance while having zero covariance with signed returns; it is not automatically that instrument. No validated external surprise is available here. [Stock and Watson (2018)](https://www.princeton.edu/~mwatson/papers/Stock_et_al-2018-The_Economic_Journal.pdf). Multi-regime IH is another feasible extension, conditional on sufficient counts, variance rank and stable loadings. These alternatives have different requirements; daily local projections alone do not solve causal identification.

A calibrated probability of termination would additionally require a fixed horizon, a verifiable termination definition, multiple conflicts or enough resolved episodes, and out-of-sample calibration. A single ongoing, right-censored conflict cannot supply that evidence by itself.
''',code='''lp=pd.read_csv(OUT/'duration_local_projections.csv')
display(lp.loc[lp.horizon.ge(0),['outcome','feature','horizon','coefficient','se','p_value','q_value','n','unit','first_date','last_date']])
display(Image(filename=str(OUT/'study_alternative.png')))
display(lp.loc[lp.horizon.eq(-1),['outcome','feature','coefficient','p_value','q_value','n','unit']])
assert len(lp.loc[lp.horizon.ge(0)])==48
assert diagnostics['lp_primary_q_lt05']==0
assert diagnostics['lp_placebo_q_lt05']==3
''',display='alternative')
    add('Trump posts: platforms, sentiment and coverage',r'''
The public [CNN Truth Social archive](https://ix.cnn.io/data/truth-social/truth_archive.json) supplies original-post links and timestamps. The January 1–September 18 New York window contains **5,743 archive records**, including 1,908 with empty readable text. Literal Iran-related retrieval yields **404 candidates and 389 distinct text strings**. Every candidate ID, text and date is checked against the raw snapshot; nine selected IDs are independently corroborated through the American Presidency Project. No complete 2026 X/Twitter corpus was obtained. Platform coverage is explicit: the empirical corpus is Truth Social.

The candidate set includes 43 repost cues, four attribution-prefix cases, three URL-only records and 17 records containing media. Eleven match a keyword only inside a URL. Counts overlap and are not additive exclusions. Repeated text at different IDs remains a publication record. Untranscribed media, implicit references, linked content, deleted posts and source gaps prevent claiming all Iran-related Trump posts.

All candidates receive timing and nonexclusive literal subject flags after URL removal: 162 military/strike, 122 shipping/blockade, 111 negotiation/agreement, 85 nuclear, 63 energy-price and 15 ceasefire/truce records. These are corpus counts, not independent events, verified claims or numerical sentiment scores. Attack and negotiation language can coexist.

The sentiment analysis is qualitative and target-specific. Praise of a conversation evaluates the conversation; a conditional threat is not an observed attack; a claimed cancellation does not establish permanent peace. Interpretation preserves speaker, target, negation, tense, quoted/reposted content, emphasis and conditions. The linked readings illustrate these distinctions. No numerical evaluation of an official or policy, individual-post causal effect, or calibrated end-of-war probability is asserted. Full row-level timing and topic audits remain local; public outputs contain aggregates and selected examples.
''',code='''audit=json.loads((OUT/'social_timing_summary.json').read_text())
display(pd.DataFrame(audit['equity_state_counts'].items(),columns=['Equity state','Records']))
print(json.dumps(audit,indent=2))
display(pd.read_csv(OUT/'social_monthly.csv'))
social_readings='''+repr(SOCIAL_EXAMPLES)+'''
social_examples=pd.DataFrame(social_readings,columns=['Date ET','Post ID','Contextual interpretation'])
social_examples['Original source']='https://truthsocial.com/@realDonaldTrump/'+social_examples['Post ID']
display(social_examples[['Date ET','Contextual interpretation','Original source']])
''',display='social')
    add('Lessons from all seven local social-text readings',r'''
The seven supplied readings inform the social-text treatment below. Short posts require contextual reading: finance polarity, generic emotional valence and conflict persistence are different targets. No dictionary coefficient or predictive accuracy from those studies is transported into this corpus without validation. Reposts, negation, sarcasm, media and changing platform coverage remain explicit limitations.
''',code='''reading_notes='''+repr(READINGS)+'''
display(pd.DataFrame(reading_notes,columns=['Authors','Reference','Application']))
''',display='readings')
    add('Investment decision and reproducibility',r'''
The analysis supports monitoring oil-supply exposure and the yield curve, but **does not support a standalone leveraged directional trade based on this news signal**. Observed oil appreciation and equity gains are historical outcomes, not forecasts. The moment restrictions, unstable oil responses, source concentration and prior-return associations prevent treating the fitted coefficients as a reliable trading rule.

For a cash allocation awaiting stronger evidence, short Treasury bills are a transparent comparison asset; the September 15 three-month CMT observation of **4.11% annualized** is a dated indicative yield, not a guaranteed investment return. Decisions to hedge an existing energy exposure should depend on that exposure, costs and horizon. A Brent–WTI position additionally carries basis, maturity and execution risks; the physical spot spread is not itself a directly executable matched-contract trade.

The notebook recomputes the three tables, positive-yield scenario, nine sensitivity checks, directional comparisons, phrase audit, timing analysis and local projections from cached inputs. Source manifests record retrieval times, actual endpoints and SHA-256 hashes; source vintages remain local. Public code provides acquisition and analysis commands. Raw articles, post records, market datasets and derived datasets are excluded from Git; the executed notebook retains aggregate tables, figures and selected linked examples. The PDF is a separate submission artifact.

The strongest remaining limits are retrospective regime thresholds; one-outlet primary coverage; no independently labeled text evaluation; revisable publisher bodies and curve estimates; nonsynchronous daily markets; limited cessation/zero-news observations; multiple war and macroeconomic channels; and incomplete social-media coverage. The estimates are reproducible conditional measurements with these limits, rather than a verified probability that the war ends soon.
''',code='''for name,digest in diagnostics['input_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
assert len(t1)==35
assert t2.available.sum()==19
assert int(t2.weak_reference_variance_shift.eq(True).sum())==2
print('Input hashes, table counts and scenario identities verified.')
''')
    return s
