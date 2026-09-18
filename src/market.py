"""Public daily market observations with immutable response caches and interval audit.

FRED oil observations are spot prices; Yahoo CL=F/BZ=F are separate futures
robustness series. No calendar filling, price substitution, or modelled data.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import time
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
START = "2025-12-15"
CUTOFF = "2026-09-16"
SERIES = ["DCOILWTICO", "DCOILBRENTEU", "DGS2", "DGS10", "DGS3MO", "T10YIE", "BAMLC0A4CBBB", "BAMLH0A0HYM2", "DTWEXBGS"]
SYMBOLS = ["CL=F", "BZ=F", "^GSPC", "^IXIC", "^VIX", "DX-Y.NYB", "EFA", "EEM", "GLD"]
SPECS = [
    ("wti_spot", "DCOILWTICO", "log_percent", "WTI Cushing spot crude oil"),
    ("brent_spot", "DCOILBRENTEU", "log_percent", "Brent Europe spot crude oil"),
    ("wti_futures", "CL=F_close", "log_percent", "Yahoo WTI futures (rolling contract)"),
    ("brent_futures", "BZ=F_close", "log_percent", "Yahoo Brent futures (rolling contract)"),
    ("sp500", "^GSPC_close", "log_percent", "S&P 500 price index"),
    ("nasdaq", "^IXIC_close", "log_percent", "Nasdaq Composite price index"),
    ("vix", "^VIX_close", "log_percent", "Cboe VIX volatility index"),
    ("dollar", "DX-Y.NYB_close", "log_percent", "ICE U.S. Dollar Index via Yahoo"),
    ("efa", "EFA", "log_percent", "iShares MSCI EAFE ETF adjusted price, USD"),
    ("eem", "EEM", "log_percent", "iShares MSCI Emerging Markets ETF adjusted price, USD"),
    ("gold_gld", "GLD", "log_percent", "SPDR Gold Shares ETF adjusted price, USD (gold proxy)"),
    ("two_year", "DGS2", "basis_points", "2-year Treasury constant maturity yield"),
    ("ten_year", "DGS10", "basis_points", "10-year Treasury constant maturity yield"),
    ("bill", "DGS3MO", "basis_points", "3-month Treasury constant maturity yield"),
    ("breakeven_10y", "T10YIE", "basis_points", "10-year breakeven inflation rate"),
    ("bbb_spread", "BAMLC0A4CBBB", "basis_points", "ICE BofA BBB US corporate option-adjusted spread"),
    ("hy_spread", "BAMLH0A0HYM2", "basis_points", "ICE BofA US high-yield option-adjusted spread"),
    ("broad_dollar", "DTWEXBGS", "log_percent", "Federal Reserve nominal broad US dollar index"),
]


def _now():
    return datetime.now(timezone.utc).isoformat()


def fetch_bytes(url, path, params=None):
    """Cache exact responses and original retrieval time; do not redownload a vintage."""
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    if path.exists() and sidecar.exists():
        payload = path.read_bytes()
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Cached response hash mismatch: {path}")
        if metadata.get("request_start") != START or metadata.get("request_cutoff_inclusive") != CUTOFF:
            raise ValueError(f"Cached response has a different requested window: {path}")
        return payload, metadata
    errors = []
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0 (academic research)"}, timeout=45)
            response.raise_for_status()
            payload = response.content
            if not payload:
                raise ValueError("Empty response")
            metadata = {
                "url": response.url, "retrieved_utc": _now(),
                "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                "http_status": response.status_code,
                "raw_path": path.relative_to(ROOT).as_posix(),
                "request_start": START, "request_cutoff_inclusive": CUTOFF,
                "failed_attempts": errors,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return payload, metadata
        except (requests.RequestException, ValueError) as exc:
            errors.append({"attempt": attempt + 1, "at_utc": _now(), "error": str(exc)})
            if attempt == 3:
                raise RuntimeError(json.dumps(errors)) from exc
            time.sleep(2 ** attempt)


def _coverage(frame, column):
    s = frame[column].dropna()
    return {"first": s.index.min().strftime("%Y-%m-%d") if len(s) else None,
            "last": s.index.max().strftime("%Y-%m-%d") if len(s) else None,
            "observations": int(len(s)), "observations_2026": int((s.index >= "2026-01-01").sum()),
            "missing_returned_rows": int(frame[column].isna().sum())}


def _fred(sid):
    raw = ROOT / "data/raw/market"
    payload, metadata = fetch_bytes("https://fred.stlouisfed.org/graph/fredgraph.csv", raw / f"{sid}.csv",
                                    {"id": sid, "cosd": START, "coed": CUTOFF})
    frame = pd.read_csv(io.BytesIO(payload), index_col=0, parse_dates=True)
    if sid not in frame:
        raise ValueError(f"FRED response missing {sid}")
    frame = frame[[sid]].apply(pd.to_numeric, errors="coerce").loc[START:CUTOFF]
    metadata.update(series=sid, provider="FRED", source_url=f"https://fred.stlouisfed.org/series/{sid}",
                    vintage="Latest vintage at retrieval; not a historical ALFRED release vintage",
                    unit="USD per barrel" if sid.startswith("DCOIL") else "index" if sid == "DTWEXBGS" else "percent", **_coverage(frame, sid))
    return frame, metadata


def _yahoo(symbol):
    raw = ROOT / "data/raw/market"
    start = int(pd.Timestamp(START, tz="America/New_York").timestamp())
    # Yahoo period2 is exclusive; the following local midnight includes cutoff close.
    end = int((pd.Timestamp(CUTOFF, tz="America/New_York") + pd.Timedelta(days=1)).timestamp())
    payload, metadata = fetch_bytes(f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}",
                                    raw / f"{symbol}.json",
                                    {"period1": start, "period2": end, "interval": "1d", "events": "div,splits"})
    chart = json.loads(payload)["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"Yahoo error for {symbol}: {chart.get('error')}")
    result = chart["result"][0]
    zone = result.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(zone).tz_localize(None).normalize()
    close = result["indicators"]["quote"][0]["close"]
    adjusted = result["indicators"].get("adjclose", [{"adjclose": close}])[0]["adjclose"]
    frame = pd.DataFrame({symbol: adjusted, symbol + "_close": close}, index=dates).loc[START:CUTOFF]
    if frame.index.duplicated().any():
        raise ValueError(f"Duplicate local session dates for {symbol}")
    metadata.update(series=symbol, provider="Yahoo Finance", source_url=f"https://finance.yahoo.com/quote/{quote(symbol, safe='')}/history/",
                    vintage="Yahoo adjusted close and raw close retained; adjusted ETFs can revise for corporate actions",
                    exchange_timezone=zone, yahoo_name=result.get("meta", {}).get("longName"),
                    currency=result.get("meta", {}).get("currency"), **_coverage(frame, symbol))
    return frame, metadata


def daily_changes(levels):
    """Real observation-to-observation changes, retaining each exact time interval."""
    result = pd.DataFrame(index=levels.index)
    for name, col, kind, _ in SPECS:
        s = levels[col].dropna() if col in levels else pd.Series(index=pd.DatetimeIndex([]), dtype=float)
        if kind == "log_percent":
            if (s <= 0).any():
                raise ValueError(f"Nonpositive {col}: log return undefined")
            values = 100 * np.log(s).diff()
        else:
            values = 100 * s.diff()
        prior = pd.Series(s.index, index=s.index).shift(1)
        result[name] = values
        result[name + "_prior_date"] = prior
        result[name + "_gap_days"] = (pd.Series(s.index, index=s.index) - prior).dt.days

    # Require both spot markets' own preceding observations to coincide. Merely
    # subtracting the latest available changes can otherwise mix two intervals.
    common = (result.wti_spot_prior_date.eq(result.brent_spot_prior_date)
              & result.wti_spot.notna() & result.brent_spot.notna())
    joint = levels[["DCOILWTICO", "DCOILBRENTEU"]].dropna()
    spread = joint.DCOILBRENTEU - joint.DCOILWTICO
    joint_prior = pd.Series(joint.index, index=joint.index).shift(1).reindex(result.index)
    common &= joint_prior.eq(result.wti_spot_prior_date)
    result["brent_wti_spread"] = spread.diff().reindex(result.index).where(common)
    result["brent_wti_spread_prior_date"] = joint_prior.where(common)
    result["brent_wti_spread_gap_days"] = result.wti_spot_gap_days.where(common)
    result["brent_wti_spread_same_interval"] = common
    result.index.name = "date"
    return result


def download():
    """Write raw provenance, levels, changes, and a machine-readable schema."""
    raw = ROOT / "data/raw/market"
    processed = ROOT / "data/processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    frames, metadata, failures = [], [], []
    jobs = [("FRED", sid, _fred) for sid in SERIES] + [("Yahoo", symbol, _yahoo) for symbol in SYMBOLS]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [(provider, name, pool.submit(fn, name)) for provider, name, fn in jobs]
        for provider, name, future in pending:
            try:
                frame, info = future.result()
                frames.append(frame)
                metadata.append(info)
                print(f"{name:12s} {info['first']}..{info['last']} n={info['observations']} (2026={info['observations_2026']})", flush=True)
            except Exception as exc:
                failures.append({"provider": provider, "series": name, "at_utc": _now(), "error": str(exc)})
                print(f"FAILED {name}: {exc}", flush=True)
    manifest = {"assembled_utc": _now(), "start": START, "cutoff_inclusive": CUTOFF,
                "missing_value_policy": "No imputation or forward filling", "series": metadata, "failures": failures}
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not frames:
        raise RuntimeError("All market requests failed; see raw market manifest")
    levels = pd.concat(frames, axis=1).sort_index()
    for col in SERIES:
        if col not in levels:
            levels[col] = np.nan
    levels["brent_wti_spread"] = levels.DCOILBRENTEU - levels.DCOILWTICO
    levels.index.name = "date"
    levels.to_csv(processed / "market_levels.csv")
    changes = daily_changes(levels)
    changes.to_csv(processed / "market_changes.csv", date_format="%Y-%m-%d")
    specs = [{"outcome": name, "level_column": col, "change_unit": kind, "label": label}
             for name, col, kind, label in SPECS]
    specs.append({"outcome": "brent_wti_spread", "level_column": "brent_wti_spread",
                  "change_unit": "USD_per_barrel", "label": "Brent minus WTI spot spread"})
    schema = {"created_utc": _now(), "start": START, "cutoff_inclusive": CUTOFF, "outcomes": specs,
              "prior_date": "Immediately preceding actual nonmissing observation for that series; no calendar filling",
              "log_percent": "100 * log(P_t/P_prior); raw close for indices/futures, adjusted close for ETFs",
              "basis_points": "100 * (yield_t - yield_prior), source yields in percentage points",
              "spread": "Brent spot minus WTI spot in USD/barrel; change only when both prior dates and joint prior date match",
              "regression_alignment": "Compare prior_date fields for every joint model; end-date overlap alone does not imply aligned returns",
              "raw_manifest": "data/raw/market/manifest.json",
              "processed_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in [processed / "market_levels.csv", processed / "market_changes.csv"]}}
    (processed / "market_manifest.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return levels, changes
