# FRE-GY 7871A — Assignment 3

Panagiotis Housos | ph2606 | Fall 2026

[Open the executed notebook](assignment3.ipynb) for the Iran war-risk study.
The report follows Rigobon and Sack's original uncentered method and replicates
the purposes of Tables 1–3: a dated news chronology, three IV sensitivity
estimates, and conditional variance shares. It then studies duration-related
news, market timing, and an alternative with separate cessation and fighting
information. WTI, Brent, their physical spread, ten-year yields, the yield
curve, global assets, and the economic comparison with 2003 are included.

The observation cutoff is **September 16, 2026**. Fitted off-the-run Treasury
par yields end on September 11, CMT yields and spot oil on September 15, and
most exchange-traded series on September 16. The main fit uses 23 major-news
and 23 nearest comparison sessions. Every available response has an uncertain
reference-variance shift, so the results do not establish causal hedge ratios.
None of 48 forward news coefficients survives multiple-test adjustment;
five of 16 prior-return timing checks do. Those checks matter because a daily
archive becomes available after the underlying news.

GDELT bulk acquisition covers 259 daily files, retaining Iran-linked event
records and source URLs. Its event codes are **not downloaded full article
text**. All 371,309 canonical source URLs and 1,025 publisher headlines receive
explicit categories, with mixed and unclear cases retained. All 177 sessions
have some GDELT war-coded news, so a non-major-event day is not a literal
no-news day. Publisher bodies supply context and relevance checks.
The Trump appendix covers 403 retrieved Truth Social text candidates,
qualitative interpretation, subject cues and market-session timing.
It does not claim a complete X/Twitter or media-post archive, calibrated
war-end probabilities, or numerical political sentiment ratings.

The repository contains code, the notebook with saved aggregate outputs,
focused tests, pinned dependencies, report-building sources and
[AI_USE.md](AI_USE.md). **No data files are committed.** Raw responses, article
and post datasets, generated CSV files, figures and internal notes remain
local. The PDF is a separate submission artifact.

## Reproduce

Use Python 3.12. From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
$assignmentPython = ".\.venv\Scripts\python.exe"
& $assignmentPython -m pip install -r requirements-lock.txt
& $assignmentPython -m ipykernel install --sys-prefix --name assignment3 --display-name "Python 3 (Assignment 3)"

& $assignmentPython -m pytest tests -q --basetemp=outputs/pytest_temp
& $assignmentPython scripts/01_collect_news.py
& $assignmentPython scripts/02_get_market_data.py
& $assignmentPython scripts/03_build_text_features.py
& $assignmentPython scripts/04_social_appendix.py
& $assignmentPython -m src.social_audit
& $assignmentPython -m src.social_timing
& $assignmentPython -m src.treasury_curve
& $assignmentPython -m src.gdelt_bulk
& $assignmentPython -m src.duration_news
& $assignmentPython scripts/05_analyze.py
& $assignmentPython scripts/06_build_deliverables.py
& $assignmentPython scripts/07_execute_notebook.py
```

No API credential is needed for this pipeline. Acquisition is cached and
resumable. GDELT downloads roughly 1.5 GB of compressed daily global responses
and retains the filtered Iran records plus response hashes, not every global
ZIP. Its DOC API collector in `src/gdelt_news.py` is optional; repeated rate
limits made the bulk archive the empirical source. Failed/capped requests are
audited, never interpreted as zero news. Guardian API access would need a key;
the included publisher collection uses the public archive.

The notebook recalculates publisher features, verifies social-candidate
membership, builds timing and event classifications, estimates Tables 1–3
with 1,999 within-regime bootstrap draws, performs robustness checks and
local projections, and generates six figures. Saved outputs preserve the
analyzed snapshot. A later uncached acquisition can differ at the same date
cutoff because sources revise bodies, observations and archives.
Manifests record source URLs, coverage, retrieval information and SHA256.
Snapshot assertions in the notebook deliberately expose changed inputs.

The source-linked research calendar is embedded in `src/events.py`. It is
retrospective and nonexhaustive, uses no return-magnitude selection, and maps
news to US sessions. `src/paper_replication.py` implements uncentered
second moments, no-intercept single/pooled IV and the actual all-day
denominator for Table 3. `src/iran_study.py` controls matching, units,
missing intervals, normalization and the empirical alternative.
`src/social_timing.py` uses the NYSE calendar; those cash-equity hours do not
describe Sunday oil-futures trading. GDELT bulk data use next-day archive
availability, not invented intraday publisher timestamps.

The original year-ahead oil contract, dollar gold series and on-the-run
liquidity premium are not identically reproduced. Nearby WTI, GLD and OAS
are explicitly labeled proxies; the unavailable premium stays blank.
The original 25-bp two-year decline is a conditional scale choice, not
automatically the direction of increased Iran-war risk. Variance shares and
conventional IV t-statistics do not repair weak identification.

## PDF

Install a LaTeX distribution with XeLaTeX, then run:

```powershell
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
Copy-Item -LiteralPath report.pdf -Destination Panagiotis_Housos_Assignment3.pdf
```

The shared narrative in `scripts/study_content.py` supplies both the report
and notebook. Tables and figures come from computed outputs. Cambria/Calibri
are used when available, with Latin Modern fallbacks. Submit the separate
PDF with the repository URL.
