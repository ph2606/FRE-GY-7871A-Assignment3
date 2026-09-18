"""Published global GPR benchmark, separate from the Iran-specific corpus."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
import pandas as pd
import requests
ROOT=Path(__file__).resolve().parents[1]
URL='https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls'

def build():
    path=ROOT/'data/raw/news/gpr_daily.xls'
    if not path.exists():
        response=requests.get(URL,timeout=60);response.raise_for_status();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(response.content)
    data=pd.read_excel(path)
    data['date']=pd.to_datetime(data['date'])
    data=data.set_index('date').loc['2026-01-01':'2026-09-16',['GPRD','GPRD_ACT','GPRD_THREAT','N10D']]
    data.to_csv(ROOT/'data/processed/gpr_benchmark.csv')
    (ROOT/'data/raw/news/gpr_manifest.json').write_text(json.dumps({'url':URL,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
       'checked_utc':datetime.now(timezone.utc).isoformat(),'first':str(data.index.min().date()),'last':str(data.index.max().date()),
       'n':len(data),'construct':'Published global geopolitical-risk index; not Iran-specific; latest revised vintage, not contemporaneously available daily readings.'},indent=2))
    return data

if __name__=='__main__':build()
