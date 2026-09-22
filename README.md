# FRE-GY 7871A — Assignment 3

Panagiotis Housos | ph2606 | Fall 2026

[Open the executed notebook](assignment3.ipynb) for the study of Iran war risk
and global financial markets. The primary sample is **February 28–September
18, 2026**. February 27 closes provide the prewar baseline, and March 2 is the
first post-outbreak US cash-market session. January 1–February 27 observations
provide context outside the main estimation window. The report covers WTI,
Brent, the Brent-minus-WTI spot spread, Treasury yields and the yield curve,
equities, credit, the dollar and gold, with an economic comparison to 2003.

The main high/low classification uses **unsigned NLP news intensity** from
relevant Guardian articles per calendar day in each close-to-close window.
Intensity strictly above the full-window 75th percentile, **5.25 articles per
calendar day**, identifies **35 high-news and 105 low-news sessions**. There
are 869 relevant articles in these active-conflict windows. The threshold is
retrospective and uses no market returns. Low-news means lower retrieved
coverage; it does not establish the absence of war news.

Available fitted Treasury yields and aligned return intervals leave **34
matched H/L pairs**, with additional paired deletion where an outcome is
missing. Tables 1–3 follow Rigobon and Sack's original uncentered method:
a dated high-news chronology, three IV sensitivity estimates and conditional
variance calculations. **Two of 19 available outcome rows**, BBB and
high-yield OAS, have uncertain positive reference-variance shifts. A positive
shift in the other rows does not establish causal identification: the panel
rank-one restriction fits poorly, and stable nuisance shocks and loadings
remain assumptions. The GDELT duration-news extension has no adjusted
q-value below 0.05 among 48 forward coefficients; three of 16 prior-return
timing checks do. These results do not establish causal hedge ratios or
validated trading forecasts.

## Data and timing

The data were frozen before September 21 trading. The declared cutoff is
September 18, the latest completed US cash-market session at collection;
the study was prepared September 21. Actual source endpoints differ:

| Inputs | Last observed date in the downloaded snapshot |
| --- | --- |
| Exchange-traded series and ten-year breakeven inflation | September 18, 2026 |
| CMT Treasury yields and BBB/high-yield OAS | September 17, 2026 |
| WTI and Brent spot oil | September 15, 2026 |
| Fed broad dollar and GSW fitted off-the-run par yields | September 11, 2026 |

Missing observations are not filled. Every main daily response must share
the current and preceding observation dates with its reference yield.
The main estimates use simple percentage price returns, percentage-point
yield/OAS changes and dollar-per-barrel oil/spread changes. The separate
`market_changes.csv` acquisition output stores log percentage returns and
basis-point changes; `src/iran_study.py` constructs the paper's units directly
from levels. The original 25-bp two-year decline is a conditional scale
choice, represented as −0.25 percentage points. A separate +25-bp scenario
uses the same loadings with reversed signs.

The publisher corpus contains 1,035 headline/body documents, of which 926
meet the January–September Iran-war relevance rule. Live/multimedia and
non-news items are excluded. Publication and modification clocks remain
explicit, and the later effective clock maps articles to scheduled cash
closes. This single-outlet, heuristic corpus is not a global news census.
Loughran–McDonald lexical features describe eligible unquoted physical
conflict/economic text after political-subject and attribution exclusions;
they do not determine the main high/low split. Missing eligible text remains
missing rather than being labeled neutral.

**GDELT is a robustness source and a duration-news extension.** The 261
daily bulk files retain Iran-linked event records and 372,908 canonical
source URLs. Empirical counts require an Iranian actor, a recent event and
eligible CAMEO codes. Those codes and links are not downloaded full article
text. Their conservative next-day archive-availability clock is distinct
from publisher timestamps. Failed or capped requests are audited rather
than interpreted as zero news.

The separate phrase audit examines operation names, negation, quotation and
the targets of words such as “extend” and “end.” In the same 1,035-headline
corpus, its explicit narrow comparator retrieves 244 documents and the
expanded rule 642, adding 398 candidates. This headline-only comparison is
separate from the main headline/body relevance rule. Synthetic stress cases
check software behavior; no human-labeled precision, recall or sentiment
accuracy is claimed.

The Trump appendix audits **404 Truth Social text candidates from January
1–September 18**, retrieved from the CNN archive. It includes literal subject
and claim cues, qualitative interpretation, repost/attribution checks and
market-session timing. The set is not a complete Iran-post census: media,
implicit references, deletions and archive gaps remain. No verified complete
X/Twitter archive was acquired. Cash-equity calendars do not describe Sunday
oil-futures trading, and daily closing prices cannot isolate intraday or
overnight post responses. No numerical political sentiment ratings or
calibrated war-end probabilities are produced.

The original year-ahead oil contract, dollar gold series and traded
on-the-run liquidity premium are not identically reproduced. Nearby WTI
futures, GLD returns and credit OAS are labeled proxies. Nearby futures
include contract-roll effects; they are not a fixed 12-month forward price.
The unavailable liquidity premium stays blank. GSW coupon-equivalent par
yields are fitted off-the-run estimates; CMT yields provide a robustness
comparison.

## Reproduce

Use Python 3.12; `python` below must select that interpreter. From the
repository root in PowerShell:

```powershell
python -m venv .venv
$assignmentPython = ".\.venv\Scripts\python.exe"
& $assignmentPython -m pip install -r requirements-lock.txt
& $assignmentPython -m ipykernel install --sys-prefix --name assignment3 --display-name "Python 3 (Assignment 3)"

& $assignmentPython scripts/01_collect_news.py
& $assignmentPython scripts/02_get_market_data.py
& $assignmentPython -m src.treasury_curve
& $assignmentPython scripts/03_build_text_features.py
& $assignmentPython scripts/04_social_appendix.py
& $assignmentPython -m src.social_audit
& $assignmentPython -m src.social_timing
& $assignmentPython -m src.gdelt_bulk
& $assignmentPython -m src.duration_news
& $assignmentPython -m src.news_regimes
& $assignmentPython -m src.phrase_audit
& $assignmentPython -m pytest tests -q --basetemp=outputs/pytest_temp
& $assignmentPython scripts/05_analyze.py
& $assignmentPython scripts/06_build_deliverables.py
& $assignmentPython scripts/07_execute_notebook.py
```

No API credential is needed for this pipeline. Acquisition is cached and
resumable; public endpoints must remain accessible. GDELT downloads roughly
1.5 GB of compressed daily global responses and retains filtered Iran
records plus response hashes. Its optional DOC API collector in
`src/gdelt_news.py` encountered rate limits; the empirical comparison uses
bulk archives. Publisher collection uses public archive pages rather than
the credentialed Guardian API.

The notebook recalculates text features and audits, builds the NLP regimes,
estimates the main tables with 1,999 independent within-regime bootstrap
draws, and regenerates robustness checks, local projections and figures.
Source manifests and diagnostics record URLs, observation coverage,
retrieval metadata and SHA256 hashes. Saved notebook outputs show the
analyzed snapshot; a fresh download need not reproduce it exactly.
Assertions check consistency with the current analysis inputs. Bootstrap
intervals and conventional IV t-statistics are supplementary diagnostics,
not weak-IV-robust confidence sets or tests of every identifying assumption.

`src/news_regimes.py` defines the primary regime. `src/paper_replication.py`
implements uncentered second moments, no-intercept single/pooled IV and the
actual all-day denominator for Table 3. `src/iran_study.py` controls matching,
units and empirical comparisons. The retrospective, nonexhaustive calendar
in `src/events.py` is a separate robustness chronology. News direction is
an extension with mixed and unclear categories retained; market returns do
not assign its labels.

The repository contains code, the notebook with saved aggregate outputs,
focused tests, pinned dependencies, report-building sources and
[AI_USE.md](AI_USE.md). **No datasets are committed.** Raw responses, article
and post tables, generated CSV files, figures and internal notes remain
local and are excluded by `.gitignore`. The PDF is a separate submission
artifact.

## PDF

After building the deliverables, use a LaTeX distribution with XeLaTeX:

```powershell
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
Copy-Item -LiteralPath report.pdf -Destination Panagiotis_Housos_Assignment3.pdf
```

The shared narrative in `scripts/study_sections.py`, supporting material in
`scripts/study_content.py`, and renderer in `scripts/study_document.py`
supply the report and notebook. Tables and figures use computed outputs.
Cambria/Calibri have Latin Modern fallbacks. Submit the separate PDF with
the repository URL.
