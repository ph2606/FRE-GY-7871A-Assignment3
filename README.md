# FRE-GY 7871A — Assignment 3

Panagiotis Housos | ph2606 | Fall 2026

[Open the executed notebook](assignment3.ipynb) for the Iran war-risk study:
WTI and Brent trends and their spot spread; global asset responses; the
Rigobon–Sack heteroskedasticity method; news attention and physical-conflict
sentiment; identification checks; and an explicit investment decision.
The appendix analyzes retrieved Truth Social communication qualitatively and
audits coverage, attribution and the distinction from X/Twitter.

The frozen observation cutoff is September 16, 2026. Spot oil and Treasury
yields are available through September 15 in this snapshot, and the broad
dollar through September 11. Prices and news bodies can be revised later.
The study does not establish a stable causal war-risk hedge ratio or a
profitable text-timing strategy. The proposed 10% bill allocation is an
illustrative risk-budget decision, not an optimized backtest result.

This repository contains the notebook with saved outputs, acquisition and
analysis code, focused tests, environment files, report-building code and
[AI_USE.md](AI_USE.md). **No data files are committed.** Raw responses,
article/post text, generated CSV files, figures and internal notes remain in
ignored local directories. The PDF is a separate submission artifact.

## Reproduce

Use Python 3.12. From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
$assignmentPython = ".\.venv\Scripts\python.exe"
& $assignmentPython -m pip install -r requirements-lock.txt
& $assignmentPython -m ipykernel install --sys-prefix --name assignment3 --display-name "Python 3 (Assignment 3)"

& $assignmentPython -m pytest tests -q
& $assignmentPython scripts/01_collect_news.py
& $assignmentPython scripts/02_get_market_data.py
& $assignmentPython -m src.benchmark
& $assignmentPython scripts/03_build_text_features.py
& $assignmentPython scripts/04_social_appendix.py
& $assignmentPython -m src.social_audit
& $assignmentPython scripts/05_analyze.py
& $assignmentPython scripts/06_build_deliverables.py
& $assignmentPython scripts/07_execute_notebook.py
```

Scripts resolve paths relative to the repository; no local machine path or API
credential is required. Acquisition is cached and resumable. The first news
download may take several minutes. A public service can rate-limit requests;
the collection audit preserves failures rather than making up observations.
The dictionary is pinned to the authors' March 2026 release by SHA256.

The notebook recalculates text features, the source-membership audit,
regressions, bootstrap estimates and figures from the local inputs. The saved
outputs preserve the reported snapshot. A later uncached download may differ
even with the same observation cutoff, because the news archive, body text,
market observations and social archive are revised. Manifests record source
URLs, requested/actual coverage, hashes and retrieval information.

The source-cited event calendar is embedded in `src/events.py` as part of the
documented research design. It is retrospective, not preregistered or
exhaustive, and uses no market-return thresholds. The paper's original
uncentered moments and a centered adaptation are both implemented. Confidence
intervals are withheld where the positive anchor-variance shift is uncertain;
raw diagnostics remain visible. Variance shares are conditional calculations,
not established causal attributions.

The social appendix retrieves all literal keyword matches in the saved archive's
readable text. It does **not** claim all Iran-related posts: untranscribed media,
implicit references, linked page contents, deletions and archive gaps remain.
Truth Social is never relabeled X. No numerical official/policy sentiment
ratings or human-labeled accuracy estimates are produced.

## PDF

Install a LaTeX distribution with XeLaTeX, then run:

```powershell
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
Copy-Item -LiteralPath report.pdf -Destination Panagiotis_Housos_Assignment3.pdf
```

The report uses Cambria/Calibri where available and Latin Modern fallbacks.
Upload the separate PDF to Brightspace with the repository URL. The PDF and
generated data remain outside the GitHub submission files.
