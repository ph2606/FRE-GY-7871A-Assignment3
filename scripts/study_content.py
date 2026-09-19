"""Shared research narrative and displays for the report and executed notebook."""
from pathlib import Path
import json
import pandas as pd

SOCIAL_EXAMPLES=[
 ('2026-01-02','115824439366264186','Conditional intervention tied to treatment of protesters. Future tense and the stated condition do not establish that intervention occurred.'),
 ('2026-01-12','115884319075881590','A tariff announcement concerns trade restrictions, a separate channel from a physical interruption of crude supply.'),
 ('2026-02-04','116013105630663812','Iran is one subject of a conversation with China’s president. Favorable language describes the conversation and relationship, rather than an observed end to conflict.'),
 ('2026-03-01','116152251973821428','A conditional response to an anticipated Iranian action. Preserve the condition, future tense and emphasis when interpreting the statement.'),
 ('2026-04-01','116329512466946656','A claimed ceasefire request, a Hormuz condition and continuing-attack language coexist. These propositions concern different outcomes and should be read separately.'),
 ('2026-05-03','116512555123589170','A shipping-operation announcement appears alongside negotiation references. Navigation access and stated conditions matter for the oil-supply channel.'),
 ('2026-08-01','117023461141824050','A claimed cancellation of a planned attack is linked to an agreement condition. It is not an unconditional statement that the conflict has ended.'),
 ('2026-09-01','117196950497702512','The account reports strikes around Hormuz and states a condition for a further response. Keep the reported claim distinct from the prospective condition.'),
 ('2026-09-07','117232304514057261','A forecast about oil and gasoline prices depends on a future war outcome. It is an attributed price claim, not an observed market effect.'),
]

READINGS=[
 ('Course presentation, undated', 'Twitter sentiment analysis overview', 'Keyword retrieval is a sampling rule; historical API instructions do not guarantee complete current archives.'),
 ('Yalin Yener, 2020', 'Step by Step: Twitter Sentiment Analysis in Python', 'Keep original text and a separate normalized field; distinguish reposts from duplicate IDs; vocabulary clouds do not validate sentiment.'),
 ('Bagheri and Islam, 2017', 'Sentiment analysis of twitter data', 'Short text, abbreviations and context challenge dictionary methods; query proportions are not accuracy estimates.'),
 ('Psomakelis et al., 2014', 'Comparing Methods for Twitter Sentiment Analysis', 'Representation and contextual features matter; their manually labeled benchmark does not establish accuracy on Iran-war statements.'),
 ('Carvalho and Plastino, 2021; online 2020', 'On the evaluation and combination of state-of-the-art features in Twitter sentiment analysis', 'Preserve negation and expressive features; evaluate feature combinations and domain transfer. Their binary benchmarks do not resolve neutral or mixed war text.'),
 ('Supplied presentation, undated', 'Exploring Differences in the Sentiment Analysis Tools using Twitter Data concerning Autism Awareness', 'Separate VADER from trained classifiers and use human validation. TN/(FP+TN) is specificity, not precision; VADER proportions are not probabilities.'),
 ('Stanford-derived course slides, undated', 'Sentiment Analysis', 'Identify the attitude holder, target and textual unit; irony and mixed targets make a generic positive/negative label unsuitable for war duration.'),
]


def sections(root):
    out=root/'outputs'
    d=json.loads((out/'study_diagnostics.json').read_text())
    t2=pd.read_csv(out/'paper_table2.csv').set_index('variable')
    t3=pd.read_csv(out/'paper_table3.csv').set_index('variable')
    lp=pd.read_csv(out/'duration_local_projections.csv')
    s=[]
    def add(title,text,code='',display=None,page=False):
        s.append(dict(title=title,text=text.strip(),code=code.strip(),display=display,page=page))
    add('Research question and principal finding',r'''
How do global financial markets respond to news about the duration and physical consequences of the Iran war? Before fighting, news changes the chance of conflict; during an ongoing war, it also changes expectations of an early halt, prolonged disruption, renewed fighting and the severity of supply losses. Those are distinct from the emotional tone of an article.

The study first implements the uncentered second-moment and instrumental-variable method in Rigobon and Sack’s *The Effects of War Risk on U.S. Financial Markets*, preserving the purposes of their Tables 1, 2 and 3. It then evaluates a duration-related news alternative. The observation window is **1 January–16 September 2026**, with actual market endpoints disclosed below. The source-cited chronology contains 25 developments on 24 US sessions; fitted Treasury availability leaves 23 major-news sessions and 23 nearest comparison sessions in the main estimation.

The observed oil movement is large: WTI spot rises **87.07%**, Brent **111.04%**, and Brent minus WTI widens from **$4.77 to $23.78 per barrel**. These full-period changes cannot be attributed entirely to the war. All 19 available sensitivity rows have an uncertain reference-variance shift; the identifying restrictions do not support a reliable causal hedge ratio. Separate news regressions also require caution: none of 48 forward-response coefficients survives the multiple-test adjustment, while five of 16 prior-return timing checks do. Markets often move before the daily news archive becomes available.

The investment implication is to keep exposure diversified and use a limited short-duration liquidity allocation, rather than treat these estimated ratios as executable oil or bond trading signals. The social appendix studies 403 retrieved Trump Truth Social text candidates, explicitly distinguishes Truth Social from X, and examines targets, conditions, attribution and publication timing without treating generic sentiment as an end-of-war probability.
''',code='''from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd()
assert (ROOT/'src').is_dir(), 'Execute from the repository root.'
sys.path.insert(0, str(ROOT))
OUT = ROOT/'outputs'
pd.set_option('display.max_rows', 60)
pd.set_option('display.max_columns', 18)
pd.options.display.float_format = '{:,.4f}'.format
required = ['market_levels.csv','offrun_yields.csv','news_documents.csv',
            'social_candidates.csv','gdelt_iran_events.csv']
assert all((ROOT/'data/processed'/p).exists() for p in required), 'Run README acquisition steps first.'
from src.text_features import build as build_text
from src.social_audit import main as audit_social
from src.social_timing import build as build_social_timing
from src.duration_news import build as build_duration
from src.iran_study import run
from src.study_figures import build as build_figures
build_text()
audit_social()
social_audit, social_calendar, social_sessions, social_summary = build_social_timing()
headlines, gdelt_sources, duration_daily, duration_summary = build_duration()
diagnostics = run(bootstrap=1999)
build_figures()
for name, digest in diagnostics['input_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
t1 = pd.read_csv(OUT/'paper_table1.csv')
t2 = pd.read_csv(OUT/'paper_table2.csv')
t3 = pd.read_csv(OUT/'paper_table3.csv')
print('Analysis recomputed from cached inputs; source hashes verified.')
''')
    add('Data and the replication boundary',r'''
The reference papers are [Rigobon and Sack (2003), NBER Working Paper 9609](https://www.nber.org/papers/w9609) and [Rigobon (2003), Review of Economics and Statistics 85(4), 777–792](https://doi.org/10.1162/003465303772815727). The first studies the period before the Iraq invasion; its broader sample contains 47 business days and 17 selected news days. Its latent factor also includes other security news. The second explains why changes in relative structural-shock variances can identify simultaneous systems, subject to stable coefficients and appropriate restrictions on common shocks. Robust standard errors alone do not provide that identification.

The main reference is the Federal Reserve’s fitted two-year off-the-run **par yield**, paired with the fitted ten-year par yield (`SVENPY02` and `SVENPY10`). These Gürkaynak–Sack–Wright research estimates exclude on-the-run and first-off-the-run notes and bonds; they are a closer conceptual match to the paper than constant-maturity Treasury yields, but are revisable model estimates. The downloaded fitted curve ends on **11 September**. [Federal Reserve curve documentation](https://www.federalreserve.gov/data/nominal-yield-curve.htm).

FRED/EIA spot oil and Treasury constant-maturity yields end on 15 September; the broad dollar ends on 11 September; most exchange-traded series end on 16 September. No missing observation is forward-filled. Every one-session paired outcome must share the current and preceding US session with the reference. Incomplete pairs are removed for that outcome, so some rows have fewer than 23 pairs. Prices use simple returns, $100(P_t/P_{t-1}-1)$; yields, breakevens and credit spreads use **percentage-point changes**; dollar oil and the physical spread use dollars per barrel. Thus 0.25 percentage points equals 25 basis points.

The original 12-month oil futures contract, spot gold dollar series and ten-year traded liquidity premium are not reproduced with identical instruments. Nearby WTI futures, GLD percentage returns and credit OAS are disclosed proxies. The liquidity-premium row remains unavailable: subtracting one modeled yield curve from another would not recover the traded premium. EFA and EEM represent USD fund returns, combining overseas equity and currency exposure. DXY and the Fed broad dollar have different baskets. These boundaries matter when comparing the tables with 2003.
''',code='''market_manifest = json.loads((ROOT/'data/raw/market/manifest.json').read_text())
display(pd.DataFrame(market_manifest['series'])[['series','provider','first','last','observations_2026']])
fitted = pd.read_csv(ROOT/'data/processed/offrun_yields.csv', index_col=0)
display(fitted.tail())
display(pd.DataFrame([
 ['Treasury yields','Fitted off-the-run par yields','GSW par yields; CMT robustness'],
 ['Oil','12-month futures, dollars','Nearby WTI dollars; WTI/Brent spot and nearby percentages'],
 ['Gold','Dollar gold price','GLD fund percentage return'],
 ['Credit','BBB/high-yield Treasury spreads','BBB/high-yield option-adjusted spreads'],
 ['Liquidity','Traded on-the-run premium','Unavailable'],
], columns=['Market','Original paper','Public-data implementation']))
''',display='data',page=True)
    add('WTI, Brent, global assets and the yield curve',r'''
WTI spot increases from $57.21 on 2 January to $107.02 on 15 September; Brent increases from $61.98 to $130.80. Define the physical spread as $S_t=P_t^{Brent}-P_t^{WTI}$ and the relative Brent premium as $100(P_t^{Brent}/P_t^{WTI}-1)$. The final dollar spread is $23.78, equivalent to **22.22% of WTI**, compared with 8.34% initially. The dollar spread ranges from −$2.45 to $25.94. A wide spread is not itself a calibrated war premium: quality, location, inventories, transport and refining demand also affect it.

On 15 September the two rolling nearby futures quotes differ by approximately $2.92, compared with the $23.78 spot difference. Those are different instruments and the rolling contracts can represent different delivery months. A tradable convergence strategy would require matched expiries and basis-risk analysis; mixing spot and futures would change the question. [EIA discussion of physical prices and contract timing](https://www.eia.gov/todayinenergy/detail.php?id=67424).

The S&P 500 gains 10.11% over its observed 2026 interval, EFA 9.94% and EEM 17.46%; GLD declines 1.64%. The two-year CMT yield rises from 3.47% to 4.67%, and the ten-year from 4.19% to 5.00%. The 10y-minus-2y curve therefore narrows from 0.72 to 0.33 percentage points: **39 basis points of bear flattening**. Yield changes are not bond returns. These trends motivate the investigation but include macroeconomic and other news; they do not measure the isolated contribution of war.
''',code='''display(Image(filename=str(OUT/'study_oil.png')))
display(Image(filename=str(OUT/'study_markets.png')))
levels = pd.read_csv(ROOT/'data/processed/market_levels.csv', index_col=0, parse_dates=True).loc['2026-01-01':]
rows = []
for col in ['DCOILWTICO','DCOILBRENTEU','^GSPC_close','EFA','EEM','GLD']:
    x = levels[col].dropna()
    rows.append([col,str(x.index[0].date()),str(x.index[-1].date()),x.iloc[0],x.iloc[-1],100*(x.iloc[-1]/x.iloc[0]-1)])
display(pd.DataFrame(rows, columns=['Series','First date','Last date','First level','Last level','Change %']))
''',display='trends',page=True)
    add('Original-paper method and Table 1: dated war news',r'''
Write the reduced form as $\Delta X_t=d w_t+u_t$. The war-factor loading in the two-year yield is normalized to one. If only the variance of the orthogonal war shock changes between major-news days $H$ and nearby comparison days $L$, with stable loadings and nuisance-shock moments, then

$$\Omega_H-\Omega_L=\delta dd',\qquad \delta>0.$$

This is a **positive rank-one restriction**, not simply a claim that markets are volatile during war. The original paper assumes zero-mean changes and uses uncentered mean squares. The implementation therefore uses uncentered moments and no intercept throughout the three IV specifications. For two-year change $x$ and outcome change $y$, define $a=E_H[x^2]-E_L[x^2]$, $b=E_H[xy]-E_L[xy]$ and $c=E_H[y^2]-E_L[y^2]$. Then $\widehat d_1=b/a$ and $\widehat d_2=c/b$. With equal group sizes, the instruments are $s_tx_t$ and $s_ty_t$, where $s_t=+1$ in $H$ and $-1$ in $L$. Pooled 2SLS uses both instruments jointly.

Table 1 performs the original chronology’s role: dates, source evidence and the information conveyed. The calendar is retrospective and nonexhaustive, selected from documented developments without filtering on return magnitude. Each distinct major-news session has $H_t=1$; other observed sessions have $H_t=0$. There are **24 H and 153 non-H sessions** in the full calendar. Each estimable H is matched without replacement to the nearest eligible non-H date by calendar distance, preferring the earlier date on a tie. This reproduces the paper’s nearby-comparison logic; it does not guarantee equal non-war volatility. The main fitted-curve window yields **23 H/L pairs**. The September 14 event is retained in the chronology but cannot enter that estimation.

The event-information labels distinguish prewar movements and negotiations, fighting, announced cessation and physical disruption. An announced ceasefire may alter expectations without establishing a durable end. No numerical end probability is inferred from these labels. The binary **major-news indicator** identifies a research regime, not the sign or magnitude of the war shock. In particular, H=0 means no event on this selected major-news list; it does not mean there was no Iran reporting that day.
''',code='''display(t1[['event_date','weekday','market_date','cash_window','description','war_risk','included_in_main_estimation','source_url']])
display(pd.read_csv(OUT/'paper_matching.csv'))
design = pd.read_csv(OUT/'binary_news_days.csv')
display(design[['major_war_news_day','any_gdelt_war_news','any_guardian_war_news']].value_counts().rename('Sessions').reset_index())
''',display='table1',page=True)
    add('Table 2: financial sensitivities under the paper’s normalization',r'''
Every effect below equals $-0.25\widehat d$: a hypothetical shock scaled to reduce the two-year fitted yield by **25 basis points**. Because $x$ is in percentage points, the multiplier is −0.25. This is a common reporting scale, not proof that an increase in Iran-war risk lowers yields. The original paper’s dollar-oil row is represented by explicitly labeled nearby WTI; spot oil percentage returns and other global assets are extensions.

Parentheses contain absolute HC1 IV t-statistics, using the no-intercept finite-sample correction. The source paper does not specify its covariance estimator in enough detail to reproduce that convention exactly; classical t-statistics are also retained in the notebook. Conventional IV t-statistics rely on adequate identification. **W marks every available row** because the 95% within-regime bootstrap interval for the reference second-moment shift includes zero. Bootstrap diagnostics use 1,999 independent resamples within H and L with a fixed seed; this is an explicitly chosen diagnostic, not a bootstrap recipe specified by the war paper or a weak-IV-robust confidence set. Unreliable ratio confidence intervals are withheld, while raw moments remain available.

The pooled ten-year response is **−0.153 percentage points**, compared with −0.26 in the 2003 paper. The two single-instrument estimates disagree even in sign. The pooled S&P response is **+4.26%**, versus −3.76% in the paper. Nearby dollar WTI is +$13.58, but the outcome-instrument ratio is about +$1,396.53 because its identifying covariance is close to zero. That explosive ratio is a failure of precision, not a plausible oil forecast. Different instruments, economic conditions, selected news and weak identification all limit comparisons with the original +$0.77 response of a **12-month** oil contract.
''',code='''columns = ['label','unit','n_high','omega1_effect','omega1_abs_t_hc1',
           'omega2_effect','omega2_abs_t_hc1','pooled_effect','pooled_abs_t_hc1','weak_reference_variance_shift']
display(t2[columns])
display(t2[['label','pooled_abs_t_classical','pooled_abs_t_hc1','availability_reason']])
display(pd.read_csv(OUT/'paper_diagnostics.csv'))
''',display='table2',page=True)
    add('Table 3: conditional variance calculations',r'''
The original Table 3 reports the L and H mean squared changes and the variance contribution implied by the combined estimate. Here $q_j=\widehat d_{pool,j}^{\,2}a$. The H-day share is $100q_j/E_H[y_j^2]$. For the entire aligned estimation window, the cumulative-movement share is $100n_Hq_j/\sum_{t\in all}y_{jt}^2$. The denominator uses **all eligible days**, not merely the selected H/L sample. Shares are percentages; the first three columns have the squared unit of the variable. Attribution cells for the reference yield and unavailable liquidity premium remain blank, following the reference table’s logic.

Under stable nuisance variance, a single changing war factor, orthogonality and serial independence of daily changes, the incremental variance can be interpreted as a lower bound on war-related variance. Those restrictions are not established here. The figures are therefore **conditional diagnostics, not identified causal variance shares**. Negative or above-100% outcomes would be displayed rather than clipped.

For the ten-year yield, the H mean square is 0.002080 pp², below the L value of 0.002243 pp². Yet the single-factor formula imposes a positive predicted contribution of 0.000540 pp². Its resulting 25.97% H share and 3.32% full-window share should not be read as established war attribution. The corresponding S&P shares are 39.87% and 7.85%. In the original paper the ten-year shares were 84.7% and 62.8%, and the S&P shares 34.2% and 19.7%. Longer windows, different instruments, different shocks and weaker restrictions all change these calculations.
''',code='''display(t3[['label','unit_squared','n_high','variance_low','variance_high',
            'predicted_variance_change','high_variance_share_pct','all_variance_share_pct','full_period_n']])
''',display='table3',page=True)
    add('Identification and sensitivity checks',r'''
The two-year H mean square is about 1.80 times its L value, but its estimated increase is only 0.001441 pp² and its bootstrap interval includes zero. A larger observed variance is not sufficient evidence that only the war factor became more variable. The six-market standardized second-moment difference has multiple positive and negative eigenvalues; its distance from the nearest positive rank-one matrix is **0.607 of its Frobenius norm**. This is a descriptive restriction check, not a formal small-sample rank test.

The table below repeats the pooled estimator using CMT yields, shifting uncertain event clocks one session later, excluding controls adjacent to major events, excluding pairs that contain scheduled FOMC dates, and separating prewar from active-conflict pairs. All comparisons preserve complete H/L pairs. The sign and magnitude of oil and equity responses change considerably. The prewar sample has only five pairs and cannot bear a strong conclusion. A different source clock or nearby control is consequential, rather than a cosmetic choice.

These diagnostics make heteroskedasticity useful as the required replication and a testable economic framework, but **not the preferred stand-alone causal measure in this sample**. Prewar uncertainty, ongoing fighting, energy interruptions, negotiations and monetary news plausibly change several variances and transmission channels simultaneously. Rigobon’s general framework emphasizes stable coefficients and common-shock treatment; adding regime labels alone does not solve a lack of independent identifying information.
''',code='''rank = json.loads((OUT/'paper_rank.json').read_text())
print('Eigenvalues:', np.round(rank['eigenvalues'], 4))
print('Relative rank-one residual:', rank['rank_one_residual'])
display(Image(filename=str(OUT/'study_identification.png')))
robust = pd.read_csv(OUT/'paper_robustness.csv')
display(robust.loc[robust.variable.isin(['ten_year','sp500','wti_spot','brent_spot']),
                  ['specification','label','n_high','pooled_effect','weak_reference_variance_shift']])
''',display='robust',page=True)
    add('Why the 2026 economy can produce a different table',r'''
The 2003 sample followed the dot-com collapse, September 11, corporate-accounting concerns and weak capital spending. It was a fragile recovery after the March–November 2001 recession, rather than a recession dated to the whole 2003 window. The [Federal Reserve’s February 2003 report](https://www.federalreserve.gov/boarddocs/hh/2003/february/ReportSection1.htm) records a 1.25% federal funds target and economic slack; [NBER’s chronology](https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions) supplies the recession dates. Growth concerns and demand for safe assets provide an economic interpretation of the original negative yield and equity responses; the paper itself does not separately identify those mechanisms.

The US petroleum position is materially different. Annual crude production rises from **5.649 million barrels/day in 2003 to 13.662 in 2025**, an increase of 141.8%; June 2026 is separately observed at 13.792. Net imports of crude **and petroleum products** change from 11.238 million barrels/day in 2003 to net exports of 2.848 in 2025. As a share of products supplied, these are +56.1% and −13.7%, respectively. The US still has crude-only net imports: 1.309 million barrels/day in June 2026. These are different trade concepts. Sources: [EIA annual production](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=MCRFPUS2&f=A), [net total petroleum imports](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=MTTNTUS2&f=A), [products supplied](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=MTTUPUS2&f=A), [monthly crude net imports](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=MCRNTUS2&f=M) and [monthly production](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=MCRFPUS2&f=M).

A larger domestic producer sector can offset some consumer and importing-firm losses, but it does not insulate US consumers from global oil prices. That is a plausible mechanism, not an estimated decomposition of the positive equity coefficient. EIA’s pre-conflict benchmark puts first-half 2025 Hormuz flows at 20.9 million barrels/day, about 20% of global petroleum-liquids consumption and one-quarter of maritime oil trade. These are **2025 exposure figures**, not September 2026 traffic estimates. [EIA prices and world-market exposure](https://www.eia.gov/energyexplained/oil-and-petroleum-products/prices-and-outlook.php); [World Oil Transit Chokepoints](https://www.eia.gov/international/content/analysis/special_topics/World_Oil_Transit_Chokepoints/).

Inflation and the expected rate path can also change the yield response. January 2003 headline/core CPI inflation was 2.6%/1.9% year over year; August 2026 was 3.4%/2.4%. August energy and gasoline prices were 16.3% and 27.4% above a year earlier. These observations support examining an energy-inflation channel without attributing all inflation to war. [BLS January 2003](https://www.bls.gov/news.release/archives/cpi_02212003.pdf); [BLS August 2026](https://www.bls.gov/news.release/archives/cpi_09112026.htm). BEA’s second estimate puts Q2 2026 real GDP growth at 1.5% at an annualized quarterly rate, following 2.1% in Q1, with private domestic final sales up 4.2% annualized. These are not year-over-year rates. [BEA release](https://www.bea.gov/news/2026/gdp-second-estimate-and-corporate-profits-2nd-quarter-2026).

The public CMT curve on 14 March 2003 was 1.56% at two years and 3.72% at ten years, a 2.16-point slope; the 15 September 2026 slope is only 0.33 points. This compares descriptive public curves, not identical fitted regression inputs. [Historical H.15 release](https://www.federalreserve.gov/releases/h15/20030317/). The 2026 bear flattening is consistent with a changed expected short-rate path, but ten-year yields also contain real-rate expectations, inflation compensation and term/liquidity premia. This analysis cannot separate those components.

The 16 September FOMC announcement at 14:00 ET raised the target range by 25 bp to 3.75–4.00%, effective 17 September. It is a scheduled macroeconomic confound inside the broad market window. The frozen CMT and fitted-curve snapshots end earlier, so their missing September 16 response cannot be inferred. [FOMC statement](https://www.federalreserve.gov/newsevents/pressreleases/monetary20260916a.htm); [implementation note](https://www.federalreserve.gov/newsevents/pressreleases/monetary20260916a1.htm). Different coefficients can consequently reflect economic exposure, news composition, instrument maturity, normalization and identification failure. The table alone cannot determine their relative contributions.
''',page=True)
    add('Free news connectors and duration-related language',r'''
**GDELT is the main free connector for the full historical window.** Its keyless DOC endpoint supports article discovery but can cap results at 250 and returned repeated rate limits during collection. The complete empirical feed instead uses GDELT’s public daily bulk event archive: all **259 daily files** for 1 January–16 September returned successfully. Iran actor/action-location filtering retains 1,807,433 event rows and 371,309 canonical source URLs. These are event records and linked sources, **not 371,309 downloaded article bodies**. Repeated actor-event rows are collapsed to one source URL per availability session. [GDELT data access](https://www.gdeltproject.org/data.html); [event data codebook](https://data.gdeltproject.org/documentation/GDELT-Data_Format_Codebook.pdf).

The Guardian’s free academic/noncommercial API is useful for full-text research, but needs a key and has a 500-request daily allowance. No key was available; the supplementary text corpus was obtained from the public publisher archive, not misdescribed as an API download. NewsAPI’s free developer tier offers 100 requests/day, a 24-hour delay and one month of searchable history, so it does not cover the entire January–September window. [Guardian access terms](https://open-platform.theguardian.com/access/); [NewsAPI plans](https://newsapi.org/pricing). Bulk GDELT is therefore the practical primary free source here, with publisher text as a context check.

The publisher collection discovers 1,893 URLs. Excluding 444 live/multimedia pages, 416 non-news pages and four failed downloads leaves 1,029 extracted documents; four bodies modified after the cutoff are excluded, leaving **1,025 documents, 917 war-relevant**. Relevance uses Iran and conflict/energy context in the headline or at least two body sentences. Classification then uses the headline, while preserving the full-body provenance and the later of publication/modification time. These retrieved bodies cannot reconstruct every original real-time article.

Every eligible publisher headline and every retained GDELT source URL receives a category. The duration distinction is operationalized as **cessation/reopening evidence versus continuing fighting/disruption**, with mixed, negotiation-only, attributed-statement and unclear cases explicit. Words suited to this conflict include Hormuz, tanker, blockade, shipping, reopening, missile, drone, strike, ceasefire and truce. Negation matters: “ceasefire collapses” differs from “ceasefire extended”; extending a ceasefire is not extending the war. Illustrative phrases explain rules, not additional observed news.

GDELT CAMEO 087 and descendants describe de-escalation, including declared/observed ceasefires, easing blockades, demobilization and retreat; roots 19/20 describe fighting/violence. Only records whose coded event date is zero to two days before archive entry enter daily counts. Older, future-dated or unknown-age mentions remain auditable but do not enter that signal. A source with both code types belongs to the mixed category and can contribute to both daily counts. [CAMEO manual](https://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf). Code 0871 can describe a declaration, so it cannot certify that hostilities ended. Iran-linked domestic violence, extraction errors, repeated wire stories and uneven English-language coverage can contaminate these measures.

Binary variables record **whether qualifying news is present**: major-war-news day, any retrieved war news, cessation-coded news and fighting-coded news. All 177 observed sessions contain some GDELT war-coded source material; the broad feed therefore provides no strict zero-war-news comparison. The Guardian has 20 zero-article sessions, which are outlet-level zeros only. Missing downloads would be missing coverage, not zero news. The paper’s H/L design uses major versus other news days instead.

This is target-specific interpretation of sentiment around conflict, not a calibrated probability that the war ends soon. An actual probability would require a fixed horizon, a verifiable termination definition, independent labels and out-of-sample calibration. One ongoing, right-censored conflict cannot supply those labels by itself. Generic positive language about a successful attack can coexist with continued fighting; negotiations need not mean cessation. No accuracy estimate or probability calibration is claimed.
''',code='''print(json.dumps(duration_summary, indent=2))
display(pd.DataFrame(duration_summary['guardian_categories'].items(), columns=['Publisher headline category','Items']))
display(pd.DataFrame(duration_summary['gdelt_categories'].items(), columns=['GDELT source category','URLs']))
coverage = pd.read_csv(OUT/'gdelt_bulk_coverage.csv')
assert len(coverage)==259 and coverage.status.astype(str).eq('200').all()
display(duration_daily.describe())
display(Image(filename=str(OUT/'study_news.png')))
''',display='news',page=True)
    add('Friday–Sunday news, market hours and next-session responses',r'''
The reference clock is the NYSE core session, **09:30–16:00 New York time**, using the actual 2026 holiday calendar and daylight-saving conversion. A pre-open or intraday item maps to the first scheduled close strictly after its timestamp; an item at or after the close maps to the next session. Friday before close can affect Friday; Friday after close, Saturday and Sunday usually map to Monday, or the next trading day after a holiday. A separate first-open clock and the following reaction session are retained. [NYSE hours and calendars](https://www.nyse.com/trade/hours-calendars).

For example, the February 28 Saturday outbreak maps to March 2; the April 7 announcement at 18:32 ET maps to April 8; the April 12 Sunday news maps to April 13. Known report times establish availability, not necessarily the first occurrence of the underlying event. Other date-only developments have uncertain intraday placement and are shifted one session in the sensitivity analysis. The reaction-window audit reports both the first mapped session’s daily change and the following session’s change, including explicit Friday–Sunday flags.

The publisher sample has 368 Friday–Sunday items and the Truth Social set 161. Of the latter, 128 appear during the equity session, 81 before open, 82 after close, 101 on weekends and 11 on holidays; 15 are Friday after-close posts. The two sources use different clocks: the publisher’s later publication/modification time and the post’s original timestamp. Neither necessarily equals first public information, particularly for reposts.

Cash-equity closure is not closure of every market. Standard WTI futures reopen Sunday evening at 18:00 ET and trade nearly around the clock with a daily break. Treasury CMT observations are based on indicative bids around 15:30 ET, while US equities close at 16:00. [CME WTI contract overview](https://www.cmegroup.com/education/courses/event-contracts-underlying-markets/wti-overview); [Treasury rate methodology](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics). Daily labels cannot recover exact synchronous intraday responses, and this panel does not contain the opening quotes needed to separate overnight from intraday returns.

GDELT bulk dates provide daily archive information, not publisher timestamps. The conservative availability clock is **07:00 ET on the next calendar day**, then the first subsequent US close; this is no earlier than the archive’s documented next-morning availability. Consequently the empirical news model studies archive-availability associations. It must not silently assign that file to a tradable previous-day news signal.
''',code='''display(Image(filename=str(OUT/'study_timing.png')))
reactions = pd.read_csv(OUT/'event_reaction_windows.csv')
display(reactions.loc[reactions.friday_sunday & reactions.outcome.eq('sp500'),
 ['event_date','cash_window','reaction_session','following_session','reaction_change','following_change','unit']])
display(reactions.loc[reactions.event_date.eq('2026-04-07'),
 ['event_date','outcome','reaction_session','following_session','reaction_change','following_change','unit']])
display(pd.DataFrame(social_summary['equity_state_counts'].items(), columns=['Equity state','Posts']))
''',display='timing',page=True)
    add('An alternative suited to ongoing conflict',r'''
A single war factor can combine expected duration, supply damage and risk aversion. The implemented alternative retains **two separate physical-news dimensions**, cessation-coded and fighting-coded source counts, rather than subtracting one from the other or forcing every article into a binary forecast. On the US session grid define $z_t^k=\Delta\log(1+N_t^k)/sd[\Delta\log(1+N_t^k)]$ for each category $k$. Scaling is retrospective over the available sample, not an out-of-sample trading calibration.

For each of eight financial outcomes, estimate cumulative responses from the preceding close to horizons $h=0,1,5$ using both news variables, their lags, a lagged outcome, log calendar-interval length, month effects and scheduled FOMC indicators. The estimation begins on March 2 to focus on active conflict. Price outcomes are cumulative simple percentage returns; yield outcomes and the oil spread are differences in their stated units. HAC inference uses 5+h lags, finite-sample correction and t inference. The 48 primary coefficients form one Benjamini–Hochberg adjustment family. This controls a multiple-testing criterion under its dependence conditions; it does not remove endogeneity.

None of the 48 coefficients has adjusted $q<0.05$; the smallest q is about **0.424**. This is limited evidence in a short noisy sample, not proof that war news has no market effect. Regressing the **preceding** session’s return on the same archive-news change provides 16 separate timing checks, with a lag two outcome control to avoid putting the dependent variable itself on the right side. Five survive that family’s adjustment: WTI/fighting, Brent/cessation, Brent/fighting, S&P/cessation and S&P/fighting. Significant movements before archive availability are consistent with already-incorporated public news, common shocks or reporting that follows markets. These checks prevent interpreting the fitted coefficients as causal or immediately tradable.

The more suitable causal research design would use independently verified **first-public-event timestamps** and narrow windows in matched-maturity oil futures, Treasury prices and equity futures. Separate documented cessation, supply interruption and renewed-fighting events; retain mixed/uncertain cases; distinguish Sunday futures reactions from Monday cash reactions; control or exclude overlapping CPI, FOMC and other scheduled releases. Labels should be assigned without viewing market returns and checked by independent annotators. Multiple historical conflicts and a declared endpoint would be needed to estimate and validate termination probabilities. Complete historical intraday quotes and reliable first-public timestamps were not acquired here, so that design is a proposal, not a claimed completed experiment.

Compared with the original single-variance-shift method, this design better matches the ongoing conflict’s multiple channels and timing. Its credibility would depend on unanticipated information, precise clocks and clean windows. The daily local projections provide a transparent empirical extension and a timing diagnostic; they do not automatically outperform the paper as causal identification.
''',code='''lp = pd.read_csv(OUT/'duration_local_projections.csv')
display(lp.loc[lp.horizon.ge(0), ['outcome','feature','horizon','coefficient','se','p_value','q_value','n','unit','first_date','last_date']])
display(Image(filename=str(OUT/'study_alternative.png')))
display(lp.loc[lp.horizon.eq(-1), ['outcome','feature','coefficient','p_value','q_value','n','unit']])
assert len(lp.loc[lp.horizon.ge(0)]) == 48
assert diagnostics['lp_primary_q_lt05'] == 0
assert diagnostics['lp_placebo_q_lt05'] == 5
''',display='alternative',page=True)
    add('Trump posts: platform, coverage and interpretation',r'''
The social corpus comes from the public [CNN Truth Social archive](https://ix.cnn.io/data/truth-social/truth_archive.json), with original-post links retained. The 2026 New York date window contains 5,697 archive records; 1,894 have empty readable text and are media records. Literal Iran-related keyword retrieval produces **403 candidate records with 388 distinct text strings**. All 403 IDs, texts and dates are checked against the raw snapshot; nine selected IDs are independently corroborated through the American Presidency Project archive. No complete 2026 X/Twitter corpus was obtained. Truth Social is therefore analyzed as Truth Social, with the X coverage gap explicit.

The candidate set includes 43 explicit repost cues, four attribution-prefix cases, three URL-only records and 16 records with untranscribed media. Eleven match a keyword only inside a URL; those are retrieval candidates, not verified Iran prose. Counts overlap and are not additive exclusions. Repeated wording at different IDs remains a publication record; duplicate text does not imply duplicate IDs. Media, implicit references, linked contents, deleted posts and source gaps prevent an “all posts” claim. A post’s timestamp does not establish that it introduced new information.

All retrieved candidates receive timing and nonexclusive literal subject flags after URL removal. There are 162 records with military/strike wording, 122 with shipping/blockade wording, 111 with negotiation/agreement wording, 85 with nuclear terms, 63 with energy-price terms and 15 with ceasefire/truce wording. These are corpus counts, not sentiment scores, independent events or verified claims. Negotiation and attack language may coexist. Complete row-level audits remain local; the notebook displays aggregate results and selected linked readings.

Sentiment interpretation follows the supplied literature’s distinction between **speaker, target and proposition**. Praise of a conversation concerns the conversation; a conditional threat is not an observed attack; a claimed cancellation does not establish permanent peace. Preserve negation, future tense, quoted/reposted material, emphasis and stated conditions. The selected readings below show how those distinctions affect the economic interpretation. No causal effect of an individual post, calibrated war-end probability, or numerical evaluation of an official or policy is asserted.
''',code='''audit = json.loads((OUT/'social_timing_summary.json').read_text())
display(pd.DataFrame(audit['claim_cue_record_counts'].items(), columns=['Nonexclusive literal subject cue','Candidate records']))
print('Platform:', audit['platform'], '| Records:', audit['candidate_records'])
print('Friday–Sunday:', audit['friday_to_sunday_posts'], '| Friday after close:', audit['friday_after_close_posts'])
examples = '''+repr(SOCIAL_EXAMPLES)+'''
display(pd.DataFrame([[date,'Truth Social',f'https://truthsocial.com/@realDonaldTrump/{post}',reading]
                      for date,post,reading in examples],columns=['Date ET','Platform','Original URL','Interpretation']))
''',display='social',page=True)
    add('Lessons from all seven local social-text readings',r'''
The supplied materials jointly support transparent collection, preserving originals, careful normalization, target-aware interpretation and domain-specific validation. Generic TextBlob or VADER polarity does not measure conflict duration. Removing stopwords or punctuation indiscriminately can remove negation and emphasis. A trained model also requires a genuine labeled evaluation set; published benchmark accuracy cannot be transferred to this corpus by assertion. No human-labeled Iran-war sentiment benchmark was created in this study.

The full set of locally supplied social-media readings and their application is listed below. Bibliographic records for the research papers are [Bagheri and Islam](https://arxiv.org/abs/1711.10377), [Psomakelis et al.](https://www.scitepress.org/PublishedPapers/2014/50753/50753.pdf), and [Carvalho and Plastino](https://link.springer.com/article/10.1007/s10462-020-09895-6). The latter appeared online in 2020 and in the 2021 journal volume. The unattributed teaching presentations are identified by title; file metadata alone is not treated as proof of authorship or publication date.
''',code='display(pd.DataFrame('+repr(READINGS)+", columns=['Reference','Title','Application']))",display='readings',page=True)
    add('Investment decision, limitations and reproducibility',r'''
My decision is an **illustrative 10% allocation to a Treasury bill with about three months remaining**, retaining the diversified core. The weight is a stated risk budget, not an optimized or backtested output of the text models. At the 15 September three-month constant-maturity benchmark of 4.11%, simple three-month carry is approximately $4.11\%\times0.25=1.0275\%$ on that allocation, or **0.103 percentage points** on the whole portfolio before costs. A CMT rate is a benchmark, not an executable bill quote. A first-order duration approximation of 0.25 years implies about a 0.25% price loss for a 100-bp yield rise, or 0.025 percentage points at portfolio level; this is a mark-to-market illustration and differs from holding a bill to maturity.

Short duration reduces interest-rate exposure but is not a complete hedge against an energy-price shock. The wide physical oil spread does not establish convergence; the fragile IV estimates do not support leveraged hedge ratios; and archive-availability correlations do not justify a news-timing trade. A portfolio with a specific oil-consumption liability could require a matched physical or futures hedge, which is outside the portfolio and data specified here. No trade was placed.

The strongest empirical conclusions are descriptive market trends and documented identification limits. The calendar is retrospective and incomplete; event clocks are sometimes uncertain; data are revised snapshots; closing times differ; public proxies do not exactly reproduce proprietary 2003 instruments; GDELT codes can misclassify events; repeated coverage is not independent news; and the publisher corpus is one English-language outlet. The estimated positive variance shift is weak, the rank-one restriction fits poorly, and macroeconomic developments can change nuisance variance. None of these limitations is repaired by a larger count of articles.

The [public repository](https://github.com/ph2606/FRE-GY-7871A-Assignment3) contains acquisition code, the fully executed notebook, source-linked research design, focused tests, pinned dependencies and an AI-use disclosure. Raw articles, posts, downloaded data, generated CSV files and internal notes are not committed. Cached source manifests retain URLs, coverage, retrieval information and hashes. The notebook recomputes the classifications, timing audit, Tables 1–3, bootstrap diagnostics, alternative models and figures, and checks input hashes. A later uncached download may change a historical observation or article body even at the same cutoff; saved outputs preserve the analyzed snapshot. The PDF is a separate submission artifact.
''',code='''assert diagnostics['main_pairs'] == 23
assert diagnostics['main_weak_rows'] == diagnostics['main_available_outcomes'] == 19
assert duration_summary['guardian_items'] == 1025
assert duration_summary['gdelt_source_urls'] == 371309
assert social_summary['candidate_records'] == 403
for name, digest in diagnostics['input_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
display(pd.DataFrame(diagnostics['input_sha256'].items(), columns=['Input','SHA256']))
print('Saved results reflect the declared snapshot; all verification assertions passed.')
''',page=True)
    return s
