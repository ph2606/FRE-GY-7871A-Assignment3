"""Compose the self-contained research narrative around executable analysis."""
from pathlib import Path
import nbformat as nbf


def build(root,diagnostics,social_examples,values):
    cells=[]
    def md(s):
        for key,value in values.items():s=s.replace('%%'+key+'%%',value)
        assert '%%' not in s,'Unresolved notebook narrative placeholder'
        cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('''# Iran war risk and global financial markets in 2026
**Panagiotis Housos · ph2606**

FRE-GY 7871A — NLP and the Investment Process · Assignment 3

Observation cutoff: **16 September 2026**; source-dependent publication lags are recorded below. This notebook applies Rigobon–Sack's covariance-difference method to Iran-related news and financial markets, with NLP measurement and a separate qualitative Truth Social appendix. It contains saved outputs. Follow the README acquisition steps before a fresh execution.

The main finding is an identification limit: the measured oil trend and physical spread are substantial, but the covariance restrictions do not support a stable, precise causal war-risk sensitivity. None of the %%PRIMARY_TESTS%% estimable primary news coefficients is statistically significant. My investment decision is a **10% illustrative allocation to a Treasury bill with about three months remaining**, retaining the diversified core; I would not use these text estimates to initiate a leveraged oil or spread trade.

This is a methodological replication with disclosed public-data proxies, not an exact reconstruction of the proprietary 2003 dataset. Numerical text measures concern physical conflict and economic conditions. The public-official appendix uses neutral qualitative interpretation, not numerical official or policy ratings.''')
    code('''from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, Image
ROOT = Path.cwd()
if not (ROOT / 'src').is_dir():
    raise RuntimeError('Execute this notebook from the repository root.')
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs'
pd.set_option('display.max_rows', 60)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 160)
pd.options.display.float_format = '{:,.4f}'.format
required = ['market_changes.csv','market_levels.csv','news_documents.csv','social_candidates.csv','gpr_benchmark.csv']
missing = [p for p in required if not (ROOT/'data/processed'/p).exists()]
assert not missing, f'Run the README acquisition steps first; missing: {missing}'
print('Input files present. No live data are silently substituted during notebook execution.')''')
    md('''## 1. Recompute the text features and econometrics
The code below rebuilds the text measures from cached bodies, checks all social candidates against the raw snapshot, and regenerates the econometric tables and figures. Random seeds and bootstrap counts are explicit. The paired bootstrap assumes independence between different pairs; it is not a block-bootstrap or formal rank test.''')
    code('''from src.text_features import build as build_text
from src.social_audit import main as audit_social
from src.analysis import run, OUTCOMES, LABELS, UNITS
from src.figures import build as build_figures
features, news_daily = build_text()
audit_social()
diagnostics = run(bootstrap=1999)
build_figures()
for name, digest in diagnostics['input_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
print('Recomputed analysis and verified input hashes.')''')
    md(r'''## 2. Sample, source vintages and market definitions
The Guardian Iran-tag archive is a single English-language outlet. News/analysis in specified news sections is retained; live blogs, multimedia and opinion sections are excluded. Raw body snapshots may have been edited. The later of publication and last-modified clocks is used; bodies revised beyond the cutoff are excluded from both timing conventions and IDF fitting. Four download failures remain in the audit.

The sample starts in January, but collection is not a claim to capture every global news story. A minimum of 80 extracted body words is required; exact duplicate bodies would be excluded. Each source URL, clock and hash is retained locally. No raw articles or full social-post dataset is committed.

Prices use $100\Delta\log P$; yields, breakevens and credit spreads use basis-point changes. The Brent–WTI spot spread uses dollar changes. Every joint model requires matching current **and preceding** observation dates. No forward filling occurs. OAS can contain month-end weekend accrual observations; holiday/nontrading intervals that do not align with the anchor are excluded.

Original off-the-run yield curves become constant-maturity Treasury yields; original credit spreads become OAS; original gold becomes GLD returns. The original 12-month oil contract is not available in this public-data panel: spot and separately labeled rolling futures are proxies. EFA/EEM combine foreign equity and exchange-rate exposure in USD. DXY and the broad dollar have different baskets. The original on-the-run liquidity premium is not replicated.''')
    code('''manifest = json.loads((ROOT/'data/raw/news/manifest.json').read_text())
text_audit = json.loads((OUT/'text_diagnostics.json').read_text())
exclusions = manifest['exclusions']
download_failures = sum(count for reason, count in exclusions.items()
                        if reason.startswith('download/parse failure:'))
display(pd.DataFrame([{'step':'Discovered unique 2026 URLs','count':manifest['discovered_unique_2026_urls']},
 {'step':'Live blog or multimedia','count':exclusions.get('live blog or multimedia',0)},
 {'step':'Non-news section','count':exclusions.get('non-news section',0)},
 {'step':'Download/parse failures','count':download_failures},
 {'step':'Extracted news documents','count':text_audit['eligible_news_documents']},
 {'step':'Revision beyond cutoff','count':text_audit['modified_after_cutoff']},
 {'step':'Within-cutoff documents','count':text_audit['within_cutoff_documents']},
 {'step':'War-relevant documents','count':text_audit['war_relevant_documents']},
 {'step':'At least 20 eligible physical-context tokens','count':text_audit['numeric_text_eligible']}]))
market_manifest = json.loads((ROOT/'data/raw/market/manifest.json').read_text())
display(pd.DataFrame(market_manifest['series'])[['series','provider','first','last','observations_2026','source_url']])
assert not market_manifest['failures']''')
    md('''## 3. WTI, Brent and the spread
$S_t=Brent^{spot}_t-WTI^{spot}_t$ is measured in USD per barrel. A positive value means Brent is more expensive. Geography, quality, transport, inventories and refining demand can affect it; it is not a pure war premium. Spot and rolling-futures observations must not be mixed. Even the two rolling-futures quotes can represent different delivery months ([EIA](https://www.eia.gov/todayinenergy/detail.php?id=67424)).

The spot spread rises from $4.77 to $23.78. Endpoint price increases are 87.07% for WTI and 111.04% for Brent (simple changes); the econometric return unit is instead log percentage points. On September 15, the rolling-futures quote difference is about $2.92, a different instrument comparison.

**Investment implication.** Do not initiate a spread convergence trade merely because the spot differential is wide. A tradable spread requires matched maturities, basis-risk analysis and a separately justified entry rule.''')
    code('''summary = pd.read_csv(OUT/'market_summary.csv')
display(summary[['variable','first_date','last_date','first','last','change','unit','n','daily_sd']])
display(Image(filename=str(OUT/'figure1_oil.png')))''')
    md('''## 4. Global assets and yield trends
Equities finish above the first January observation despite intervening drawdowns. The two-year yield rises 120 bp and the ten-year 81 bp, narrowing 10s2s from 72 to 33 bp: endpoint bear flattening. Those full-period changes include other news and do not identify the war contribution. An oil-supply/inflation shock may raise yields while growth concerns lower them; the 2003 sign pattern cannot be imposed on Iran in 2026.

**Investment implication.** Keep diversification and avoid treating long-duration Treasuries as an automatic hedge against an inflationary supply interruption. The defensive allocation uses short bills.''')
    code("display(Image(filename=str(OUT/'figure2_markets.png')))")
    md(r'''## 5. News attention and sentiment around physical conflict
Relevance requires a conflict/energy context and Iran-related entity in the headline, or at least two body sentences with both contexts. The literal rules are in `src/text_features.py`. They have no independent human-labeled accuracy estimate.

Attention is $A_t=\log(1+n_t/d_t)$, where $n_t$ is the number of relevant documents in the session interval and $d_t$ its calendar-day length. It measures retrieved coverage, not war probability. Relevance and attention use the original text independently of the lexical exclusions. The numerical sample excludes whole paragraphs containing official/policy subjects or attribution cues, then removes quoted spans before sentence splitting. Retained sentences require a physical-context term and exclude personal/institutional pronouns, rhetorical questions and quotation-bearing fragments. These conservative rules omit much factual reporting too; they cannot guarantee semantic separation. Official statements and policy assessments are not numerically rated.

For category $c$, the token-weighted daily measure is $L_{ct}=100\sum_j C_{cj}/\sum_j N_j$, using only documents with at least 20 eligible tokens. No eligible text gives a missing measure, not zero or neutral sentiment. The verified March 2026 Loughran–McDonald release has 2,345 active negative and 297 uncertainty words (positive year flags only). Tokens are alphabetic strings of at least two characters. The financial dictionary is imperfect for war news and misses contextual negation and target differences.

Assignment 1's exact supplement uses $\sum_{w\in c}(1+\log tf_{wj})\log(N/df_w)/(1+\log a_j)$, where $a_j$ is tokens divided by distinct tokens in document $j$. IDF is fitted to the eligible full corpus; these retrospective weights are not an out-of-sample signal. Literal threat and action patterns are additional context counts. The diplomacy pattern remains documented but its terms are excluded by the numerical policy filter. No numerical political evaluations are produced.''')
    code('''from src.text_features import CONFLICT, PHYSICAL, IRAN, THREAT, ACT, DIPLOMACY, POLITICAL, ATTRIBUTION, ANAPHORA
display(pd.DataFrame({'rule':['Conflict/energy','Physical context','Iran entities','Threat words','Action words','Diplomacy words','Excluded subjects/policies','Excluded attribution','Excluded anaphora'],
 'literal_pattern':[x.pattern for x in [CONFLICT,PHYSICAL,IRAN,THREAT,ACT,DIPLOMACY,POLITICAL,ATTRIBUTION,ANAPHORA]]}))
display(pd.read_csv(OUT/'news_monthly.csv'))
display(features[['n_words','negative_pct','uncertainty_pct','negative_tfidf','uncertainty_tfidf']].describe())
display(Image(filename=str(OUT/'figure3_news.png')))''')
    md('''Attention covers 917 relevant articles across %%NEWS_SESSIONS%% sessions. Only %%NUMERIC_DOCS%% articles supply enough eligible lexical text, spanning %%LEXICAL_SESSIONS%% sessions. Attention rises sharply around the outbreak and later falls. The negative-vocabulary trend is %%NEG_TREND%% percentage points per month (p=%%NEG_TREND_P%%); its post-March level difference is also nonsignificant (p=%%NEG_STEP_P%%). Uncertainty vocabulary has a +%%UNC_STEP%% percentage-point step (q=%%UNC_STEP_Q%%), but no detectable time trend (p=%%UNC_TREND_P%%). These describe a selected, sparse text sample, not changing public opinion or an official's attitude.

Lower news activity does not establish lower physical risk. The global GPR benchmark includes other countries and is a revised newspaper index, not a daily real-time Iran probability. Its correlation with log attention is %%BENCH_ATT%%, versus %%BENCH_NEG%% for negative-word share, on %%BENCH_N%% common dates.

**Investment implication.** Use attention to trigger review of exposure and source developments; do not remove an energy hedge only because vocabulary or coverage becomes less negative.''')
    code('''display(pd.read_csv(OUT/'trend_tests.csv'))
display(pd.read_csv(OUT/'benchmark_correlations.csv',index_col=0))
display(pd.read_csv(OUT/'common_words.csv').head(30))''')
    md(r'''## 6. Identification and event design
Rigobon–Sack's reduced form is $\Delta X_t=d w_t+u_t$. If only the orthogonal war-factor variance changes and loadings/nuisance covariance stay stable, $\Delta\Omega=\Omega_H-\Omega_L=\delta dd'$ with $\delta>0$: a positive rank-one matrix. Heteroskedasticity-robust OLS standard errors alone do not identify a structural effect.

With the two-year-yield response normalized to one, let $a=\Delta Var(x)$, $b=\Delta Cov(x,y)$ and $c=\Delta Var(y)$. The two estimates are $d_1=b/a$ and $d_2=c/b$. Pooled IV instruments $x$ with $s x$ and $s y$, where $s=+1$ on H and −1 on L. Centered moments use within-regime demeaning and denominator n consistently; original zero-mean second moments are also estimated.

The retrospective source calendar has 25 developments on 24 sessions, selected without return filtering. Controls are unused, within 35 calendar days, in the same pre-/post-February-28 period, not event or adjacent sessions, and at/below their own month's median attention. Same-month then closest dates are preferred. Two controls have the same attention as H and two have more; labels hypothesize a variance shift rather than demonstrate it.

Weekend news maps to the next equity session. April 7's 18:32 EDT ceasefire maps to April 8. Uncertain source clocks are shifted one session in a robustness check. Body features use the first observed session with its 15:00 ET cutoff at or after the effective timestamp. Daily market closes are not fully synchronous.''')
    code('''events = pd.read_csv(OUT/'event_calendar.csv')
matching = pd.read_csv(OUT/'matching_audit.csv')
display(events[['event_date','market_date','description','timing_uncertain','source_url']])
display(matching)
print('Events:', diagnostics['event_records'], 'Distinct sessions:', diagnostics['unique_event_dates'],
      'Matched pairs:', diagnostics['matched_pairs'], 'Alternative NLP pairs:', diagnostics['nlp_pairs'])''')
    md('''## 7. Covariance-difference results
The anchor is in basis points, so multiply every response coefficient by **−25**, not −0.25, to reproduce the paper's 25-bp-yield-decline normalization. Price outcome units remain log percentage points; the oil spread is USD/barrel; yield/OAS outcomes are bp. This normalization does not establish which direction means increased 2026 Iran risk.

The bootstrap resamples matched pairs together (1,999 draws). If the anchor variance difference is nonpositive or its bootstrap interval crosses zero, reported coefficient intervals are withheld; raw estimates and raw bootstrap draws' intervals remain diagnostics. This is not a formal weak-IV test. Four unflagged outcomes use smaller 20-pair samples, so apparent identification differences also reflect sample composition.''')
    code('''het = pd.read_csv(OUT/'heteroskedasticity.csv')
display(het[['y','n_pairs','var_ratio','delta_var_x','delta_var_x_ci_low','delta_var_x_ci_high','weak_identification']])
display(het[['y','d_reference_scenario','d_outcome_scenario','d_pooled_scenario',
             'd_pooled_scenario_ci_low','d_pooled_scenario_ci_high','weak_identification']])
display(Image(filename=str(OUT/'figure4_identification.png')))
display(pd.DataFrame({'eigenvalue':json.loads((OUT/'rank_diagnostics.json').read_text())['eigenvalues']}))''')
    md('''Thirteen of 17 outcomes fail the conservative anchor-variance diagnostic. The six-market rank-one residual is 0.501, with several nonzero/negative eigenvalues; it is a descriptive diagnostic, not a formal rejection p-value. Moment estimates disagree substantially. Even unflagged pairwise estimates have wide intervals or poor moment agreement. No table is treated as a reliable causal hedge ratio.

The supplied 2003 Table 2 pooled responses under the same yield normalization were −26 bp (10y), −3.76% (S&P), +5 bp (BBB), +34 bp (HY), −0.44% (broad dollar) and +$0.77 (12-month oil futures). Proxies and units differ; those signs are not imposed on 2026.

**Investment implication.** Do not translate the raw normalized coefficients into a leveraged cross-asset portfolio. A yield decline need not represent the same economic shock in the two conflicts.''')
    code('''oil = pd.read_csv(OUT/'oil_anchor.csv')
display(oil[['y','n_pairs','d_pooled_scenario','weak_identification']])
print('Oil normalization: +10% simple WTI-futures increase = 100*log(1.10) log points; all comparisons fail the anchor diagnostic.')''')
    md(r'''## 8. Robustness and variance attribution
Robustness covers original uncentered second moments, removal of entire pairs touching FOMC decisions, shifting uncertain event dates, within-month high-attention regimes, January–May/June–September subsamples and leave-one-pair-out influence. A changing sign is evidence against relying on one convenient specification. The NLP regime rule selects on text, but article count need not equal shock variance.

The original variance calculation is $q_j=d_j^2a$; H share is $q_j/Var_H(y)$ and the full-period quantity is $n_Hq_j/[T Var_{all}(y)]$. The denominator uses the actual all-period aligned sample, not just selected controls. The cumulative-variance interpretation additionally needs serial independence. Values below are conditional algebraic diagnostics, **not established causal lower bounds**. They are not clipped into [0,100].''')
    code('''robust = pd.read_csv(OUT/'heteroskedasticity_robustness.csv')
display(robust[robust.y.isin(['wti_spot','brent_spot','brent_wti_spread','sp500','ten_year'])][
 ['specification','y','n_pairs','delta_var_x','d_pooled_scenario','weak_identification']])
display(het[['y','var_y_low','var_y_high','incremental_variance','high_variance_share','all_variance_share','variance_attribution_caution']])
influence = pd.read_csv(OUT/'event_influence.csv')
display(influence.groupby('outcome').d_pooled.agg(['min','median','max']))''')
    md('''**Investment implication.** Timing and moment sensitivity make a large trade from this factor unjustified. The original causal variance interpretation is withheld even where the arithmetic share appears plausible.''')
    md(r'''## 9. News and financial-change regressions
Estimate $\Delta Y_t=\alpha+\beta_A z(\Delta A_t)+\beta_N z(\Delta L_{Nt})+\rho\Delta Y_{t-1}+\eta\log d_t+monthFE+e_t$.

Both text features are scaled by their full-2026 standard deviations. Current **and lagged** market intervals must align with equity sessions. Missing lexical measures remain missing. HAC uses five retained-observation lags, finite-sample correction and t inference; irregular gaps complicate the dependence correction. Models require at least 40 complete observations. BBB and high-yield OAS fall below this threshold and are omitted from this regression family; both remain in the heteroskedasticity analysis. The primary Benjamini–Hochberg family has %%PRIMARY_TESTS%% estimable coefficients (2 features × %%PRIMARY_OUTCOMES%% outcomes), out of 36 planned comparisons. The models measure associations, not causal effects of text. Month effects and lagged outcomes do not eliminate all confounding.''')
    code('''reg = pd.read_csv(OUT/'news_regressions.csv')
display(reg[['outcome','feature','coefficient','se','ci_low','ci_high','p_value','q_value','n','feature_sd','r_squared']])
display(Image(filename=str(OUT/'figure5_regressions.png')))
versions = pd.read_csv(OUT/'news_regressions_all.csv')
display(versions.groupby('version').agg(tests=('p_value','size'),min_n=('n','min'),max_n=('n','max'),
    unadjusted=('p_value',lambda x:(x<.05).sum()),adjusted=('q_value',lambda x:(x<.05).sum())))''')
    md('''%%P_UNADJUSTED%% primary coefficients have unadjusted p-values below 5%; none survives adjustment (minimum q %%MIN_Q%%). Only %%REG_MIN_N%%–%%REG_MAX_N%% complete observations per model remain, all beginning March 3 after consecutive-session lexical and market alignment requirements. The descriptive news/market history still starts in January. Null results with this limited sample are not proof that conflict information never matters. The alternate uncertainty, publication-clock, next-session and FOMC-exclusion models are disclosed separately. These retrospective tests do not establish profitable sentiment timing.

**Investment implication.** Use text as a monitoring input, with limited conviction. The bill allocation rests on carry and low duration, not on an estimated text forecast.''')
    md('''## 10. Explicit investment decision
Allocate **10% of an illustrative diversified portfolio to a Treasury bill with approximately three months remaining** and retain the diversified core. This weight is an analyst risk-budget choice, not optimized or backtested. At the September 15 DGS3MO benchmark of 4.11%, approximate three-month carry is 1.0275% on the sleeve, or 0.10275 percentage points on the whole portfolio, before costs/taxes. An actual bill quote is required for implementation.

For a $100,000 illustration, a $10,000 sleeve would earn about $102.75 using that simple approximation. Assume duration 0.25 years; instantaneous price change is approximately −duration × yield change. This stress is distinct from three-month carry and is not a total-return forecast. Holding a purchased bill to maturity fixes its nominal payoff; early sale, reinvestment, inflation and opportunity cost remain risks. No trade has been placed.''')
    code('''levels = pd.read_csv(ROOT/'data/processed/market_levels.csv',index_col=0,parse_dates=True)
bill = levels.DGS3MO.dropna()
carry = bill.iloc[-1]/100*.25
stress = pd.DataFrame({'yield_shift_bp':[-100,0,100]})
stress['bill_price_effect_pct'] = -.25*stress.yield_shift_bp/100
stress['portfolio_effect_pp'] = .10*stress.bill_price_effect_pct
display(stress)
print(f'Yield benchmark {bill.index[-1].date()}: {bill.iloc[-1]:.2f}%; approximate sleeve carry {100*carry:.4f}%; portfolio {10*carry:.5f} pp.')''')
    md('''## 11. Social-media appendix: coverage and qualitative language analysis
The user includes Truth Social in the social-media scope. [Archive documentation](https://github.com/stiles/trump-truth-social-archive) identifies the updated [CNN-hosted snapshot](https://ix.cnn.io/data/truth-social/truth_archive.json). The 2026 window has 5,697 records, including 1,894 empty-text media records and 2,775 records with media. We retrieve every literal match to the stated Iran/Hormuz/etc. rule in readable text: **403 candidates**, not all Iran-related posts.

All candidate IDs, decoded texts and New York dates are checked against a separate reconstruction from the raw snapshot. This is an automated consistency audit, not independent human labels. The archive does not prove original authorship; explicit repost and attribution cues are preserved. Keyword-only relevance can occur in URLs, campaign text or multi-topic messages. Blank media posts are missing evidence, not neutral sentiment.

No verified complete 2026 X corpus was acquired; Truth Social is not called Twitter. Media, implicit references, deleted posts and unknown archive gaps prevent a complete census. The appendix describes factual subjects and qualitative linguistic context rather than numerical official/policy ratings. It does not represent public opinion or investor sentiment.''')
    code('''social_audit = json.loads((OUT/'social_audit.json').read_text())
relevance_audit = json.loads((OUT/'social_relevance_audit.json').read_text())
display(pd.DataFrame({'measure':['2026 archive records','Empty readable text','Records with media','Text candidates','Unique candidate texts',
  'Explicit RT cues','Explicit attribution prefixes','URL-only records','URL-only keyword matches','Candidates with unreviewed media'],
 'count':[social_audit['records_in_window'],social_audit['empty_text_records'],social_audit['media_records'],
          social_audit['iran_keyword_candidates'],relevance_audit['unique_candidate_texts'],
          relevance_audit['explicit_repost_marker_records'],relevance_audit['explicit_attribution_prefix_records'],
          relevance_audit['url_only_records'],relevance_audit['relevance_match_location_counts'].get('URL only',0),
          relevance_audit['records_with_unreviewed_media']]}))
display(pd.DataFrame.from_dict(relevance_audit['source_form_counts'],orient='index',columns=['records']))
display(Image(filename=str(OUT/'figure6_social.png')))
assert relevance_audit['candidate_ids_match_independent_raw_retrieval']
assert relevance_audit['all_text_and_local_dates_match_raw_snapshot']
assert not relevance_audit['sentiment_scores_or_rankings_produced']''')
    table='| New York date | Source | Qualitative reading |\n|---|---|---|\n'
    for date,post,reading in social_examples:
        table+=f'| {date} | [Original post](https://truthsocial.com/@realDonaldTrump/{post}) | {reading} |\n'
    md(table+'''
Selected original IDs were independently matched to American Presidency Project pages; its Pacific-time display must not replace original UTC/New York dates. For example, the January 2 New York post appears on its January 1 page. This corroborates examples, not archive completeness.

Two June 11 posts first describe planned strikes and later cancellation connected to negotiations. Combining them would lose intraday sequence. Favorable language about a military operation does not imply lower physical risk. Preserve attitude holder, target, condition and attribution. Only one candidate literally names WTI; none names Brent. A price forecast in a post is not a measured Brent–WTI spread or a causal market response.

**Investment implication.** Use authenticated statements to identify claims and possible news times, verify the underlying developments independently, and avoid a trade based on a single document-level polarity label.''')
    md('''## 12. All seven supplied Twitter references

| Reading | Contribution used here | Important qualification |
|---|---|---|
| *Twitter sentiment analysis overview*, unattributed, 10 pp. | Explicit retrieval, cleaning and analysis stages | Historical API examples do not establish full 2026 access |
| Yener (2020), *Step by Step: Twitter Sentiment Analysis in Python*, 18 pp. | Preserve originals, deduplicate by ID, distinguish TextBlob/VADER | Preserve negation, punctuation and emphasis; word clouds are not validation |
| Bagheri & Islam (2017), *Sentiment analysis of twitter data*, 5 pp. | Short text and query choice complicate interpretation | A neutral share is not accuracy; [arXiv](https://arxiv.org/abs/1711.10377) |
| Psomakelis et al. (2014), *Comparing Methods for Twitter Sentiment Analysis*, 8 pp. | Contextual representations and supervised validation | Benchmark accuracy does not transfer automatically; [DOI](https://doi.org/10.5220/0005075302250232) |
| Carvalho & Plastino (online 2020, volume 2021), *On the evaluation and combination of state-of-the-art features in Twitter sentiment analysis*, 50 pp. | Negation, expressive features, feature/model evaluation and domain transfer | Binary benchmarks do not establish valid neutral/mixed war labels; [DOI](https://doi.org/10.1007/s10462-020-09895-6) |
| *Exploring Differences in the Sentiment Analysis Tools using Twitter Data concerning Autism Awareness*, 13 slides | Compare actual models with labels | Slide 11's TN/(FP+TN) is specificity, not precision; scikit-learn is a library |
| Stanford-derived *Sentiment Analysis*, 81 slides | Holder, target, textual unit, discourse and domain dependence | Official communication is not public sentiment; operation praise is not a war-risk measure |

No human gold-label dataset or model accuracy is invented. The local reference review keeps page/slide pointers. The financial sentiment supplement follows Loughran–McDonald rather than silently replacing it with a generic TF–IDF implementation.

## 13. Sources and limitations
[Rigobon (2003)](https://doi.org/10.1162/003465303772815727), *Identification Through Heteroskedasticity*, and [Rigobon & Sack (2003), NBER WP 9609](https://www.nber.org/papers/w9609), *The Effects of War Risk on U.S. Financial Markets*, are the supplied core sources. The supplied paper version determines historical tables; the later Federal Reserve version is not silently substituted. [Loughran–McDonald dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/), [GPR data](https://www.matteoiacoviello.com/gpr.htm), [Guardian Iran archive](https://www.theguardian.com/world/iran), [FOMC calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm), and per-series/per-article links in the displayed metadata support the analysis.

The remaining limitations are a single news outlet; revised text and latest-vintage market data; nonexhaustive, retrospective and sometimes date-only events; nonsynchronous closes; imperfect proxies; short samples and unstable nuisance shocks; heuristic NLP without independent labels; and incomplete X/media coverage. The study neither observes calibrated war probabilities nor proves that posts cause returns. Variance shares are conditional calculations whose causal interpretation is withheld. The model checks support modest risk-taking and monitoring, not a profitable timing claim.

Code, this notebook with saved outputs, environment files, focused tests and `AI_USE.md` belong in the repository. Data, local research notes and generated CSV files remain ignored; the PDF is a separate submission file.''')
    code('''assert diagnostics['primary_news_tests'] == len(reg)
print('Primary coefficients:', len(reg), 'Adjusted p < 0.05:', diagnostics['primary_news_q_lt_05'])
print('Frozen observation cutoff:', diagnostics['cutoff'])
print('Notebook calculations completed; raw and derived data remain outside version control.')''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (Assignment 3)','language':'python','name':'assignment3'},
       'language_info':{'name':'python','version':'3.12'}})
    nbf.write(nb,Path(root)/'assignment3.ipynb')
