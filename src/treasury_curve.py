"""Federal Reserve GSW coupon-equivalent fitted off-the-run par yields."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json
import pandas as pd
import requests
ROOT=Path(__file__).resolve().parents[1]
URL='https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv'


def build():
    folder=ROOT/'data/raw/treasury_curve';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'feds200628.csv'
    if not path.exists():
        r=requests.get(URL,timeout=90);r.raise_for_status();path.write_bytes(r.content)
        (folder/'manifest.json').write_text(json.dumps({'url':URL,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
          'sha256':hashlib.sha256(r.content).hexdigest()},indent=2))
    meta=json.loads((folder/'manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
    text=path.read_text(encoding='utf-8')
    assert 'Par yield,Coupon-Equivalent,SVENPYXX' in text
    data=pd.read_csv(io.StringIO(text[text.index('Date,'):]),parse_dates=['Date']).set_index('Date')
    sample=data[['SVENPY02','SVENPY10']].apply(pd.to_numeric,errors='coerce').loc['2025-12-15':'2026-09-16']
    (ROOT/'data/processed').mkdir(parents=True,exist_ok=True)
    sample.to_csv(ROOT/'data/processed/offrun_yields.csv')
    print('GSW coverage:',sample.dropna().index.min().date(),sample.dropna().index.max().date())
    return sample


if __name__=='__main__':build()
