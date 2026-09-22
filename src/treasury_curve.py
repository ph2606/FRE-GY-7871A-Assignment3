"""Federal Reserve GSW coupon-equivalent fitted off-the-run par yields.

Use ``python -m src.treasury_curve --refresh`` to download a new vintage while
preserving existing raw data and the earlier processed snapshot.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
import shutil

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
DEFINITION_URL = "https://www.federalreserve.gov/data/nominal-yield-curve.htm"
START = "2025-12-15"
CUTOFF = "2026-09-18"
COLUMNS = ["SVENPY02", "SVENPY10"]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _parse(payload):
    text = payload.decode("utf-8-sig")
    if "Par yield,Coupon-Equivalent,SVENPYXX" not in text:
        raise ValueError("GSW response lacks the coupon-equivalent par-yield definition")
    start = next((i for i, line in enumerate(text.splitlines()) if line.startswith("Date,")), None)
    if start is None:
        raise ValueError("GSW response has no Date header")
    data = pd.read_csv(io.StringIO(text), skiprows=start, parse_dates=["Date"]).set_index("Date")
    if not set(COLUMNS).issubset(data.columns):
        raise ValueError("GSW response lacks the required 2y/10y par-yield columns")
    if data.index.duplicated().any() or data.index.isna().any():
        raise ValueError("GSW response contains invalid or duplicate dates")
    return data[COLUMNS].apply(pd.to_numeric, errors="coerce").sort_index()


def _archive(folder, old):
    key = f"retrieved_{str(old.get('retrieved_utc', 'unknown'))[:10]}_{old['sha256'][:12]}"
    target = folder / "versions" / key
    target.mkdir(parents=True, exist_ok=True)
    for source in [folder / "feds200628.csv", folder / "manifest.json"]:
        if source.exists() and not (target / source.name).exists():
            shutil.copy2(source, target / source.name)
    source = ROOT / "data/processed/offrun_yields.csv"
    if source.exists() and not (target / "offrun_yields.csv").exists():
        shutil.copy2(source, target / "offrun_yields.csv")
    return target.relative_to(ROOT).as_posix()


def build(refresh=False):
    folder = ROOT / "data/raw/treasury_curve"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "feds200628.csv"
    manifest_path = folder / "manifest.json"
    cached = None
    old = {}
    if path.exists():
        if not manifest_path.exists():
            raise ValueError("An existing GSW raw cache has no provenance manifest")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached = path.read_bytes()
        if hashlib.sha256(cached).hexdigest() != old["sha256"]:
            raise ValueError("GSW cached raw response hash does not match its manifest")
    refresh = refresh or (bool(old) and old.get("cutoff_inclusive") != CUTOFF)
    archived = _archive(folder, old) if cached is not None and refresh else old.get("archived_prior_vintage")
    errors = [] if refresh else old.get("refresh_errors", [])
    payload, metadata = cached, dict(old)
    if cached is None or refresh:
        try:
            response = requests.get(URL, timeout=60, headers={"User-Agent": "Mozilla/5.0 (academic research)"})
            response.raise_for_status()
            _parse(response.content)
            payload = response.content
            metadata = {"url": response.url, "retrieved_utc": _now(),
                        "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                        "http_status": response.status_code}
            path.write_bytes(payload)
        except (requests.RequestException, ValueError) as exc:
            errors.append({"attempted_utc": _now(), "url": URL, "error": str(exc)})
            if cached is None:
                manifest_path.write_text(json.dumps({"url": URL, "errors": errors, "status": "unavailable"}, indent=2), encoding="utf-8")
                raise RuntimeError("GSW acquisition unavailable; no values substituted") from exc
    data = _parse(payload)
    sample = data.loc[START:CUTOFF]
    if sample.dropna(how="all").empty:
        raise ValueError("GSW source contains no usable observations in the requested sample")
    processed = ROOT / "data/processed/offrun_yields.csv"
    processed.parent.mkdir(parents=True, exist_ok=True)
    sample.index.name = "Date"
    sample.to_csv(processed, date_format="%Y-%m-%d")
    metadata.update(assembled_utc=_now(), definition_url=DEFINITION_URL,
                    source_type="Federal Reserve staff research product; not an official statistical release",
                    definition="Fitted off-the-run nominal par yields; coupon-equivalent percent; on-the-run and first-off-the-run issues excluded",
                    columns=COLUMNS, start=START, cutoff_inclusive=CUTOFF,
                    raw_path=path.relative_to(ROOT).as_posix(),
                    raw_last_observation=data.dropna(how="all").index.max().strftime("%Y-%m-%d"),
                    archived_prior_vintage=archived, refresh_requested=refresh, refresh_errors=errors,
                    status="cached fallback after failed refresh" if errors else "available",
                    missing_value_policy="Actual observations only; no filling or substitution",
                    vintage="Latest staff-model vintage at retrieval; historical estimates may be revised",
                    processed_path=processed.relative_to(ROOT).as_posix(),
                    processed_sha256=hashlib.sha256(processed.read_bytes()).hexdigest(),
                    series=[{"series": c, "first": sample[c].dropna().index.min().strftime("%Y-%m-%d"),
                             "last": sample[c].dropna().index.max().strftime("%Y-%m-%d"),
                             "observations": int(sample[c].notna().sum()),
                             "observations_since_2026_02_28": int(sample.loc["2026-02-28":, c].notna().sum())} for c in COLUMNS])
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("GSW:", metadata["status"], [(x["series"], x["first"], x["last"], x["observations"]) for x in metadata["series"]], flush=True)
    return sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    build(refresh=parser.parse_args().refresh)
