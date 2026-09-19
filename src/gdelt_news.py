"""Daily public GDELT headline retrieval with explicit caps and coverage failures.

This is a reproducible collection of retrieved headlines, not all news. The
provider's first-seen clock is retained and never called a publication clock.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START = '2026-01-01'
END = '2026-09-17'  # exclusive UTC cutoff
QUERY = '(iran OR hormuz) (war OR ceasefire OR missile OR attack OR shipping) sourcelang:english'
URL = 'https://api.gdeltproject.org/api/v2/doc/doc'


def collect():
    raw = ROOT/'data/raw/gdelt/daily'
    raw.mkdir(parents=True, exist_ok=True)
    (ROOT/'data/processed').mkdir(parents=True, exist_ok=True)
    (ROOT/'outputs').mkdir(parents=True, exist_ok=True)
    records, audit = [], []
    session = requests.Session()
    last_request = 0.0
    for day in pd.date_range(START, pd.Timestamp(END)-pd.Timedelta(days=1)):
        date = day.strftime('%Y-%m-%d')
        path = raw/(date+'.json')
        meta_path = raw/(date+'.metadata.json')
        params = {'query':QUERY, 'mode':'artlist', 'format':'json', 'maxrecords':250,
                  'sort':'HybridRel', 'startdatetime':day.strftime('%Y%m%d%H%M%S'),
                  'enddatetime':(day+pd.Timedelta(days=1)).strftime('%Y%m%d%H%M%S')}
        payload = None
        meta = {'date':date,'query':QUERY,'maxrecords':250,'sort':'HybridRel','errors':[]}
        if path.exists() and meta_path.exists():
            payload = path.read_bytes()
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if hashlib.sha256(payload).hexdigest()!=meta['sha256']:
                raise ValueError(f'GDELT cache hash mismatch for {date}')
        else:
            for attempt in range(4):
                time.sleep(max(0,6.3-(time.monotonic()-last_request)))
                last_request = time.monotonic()
                try:
                    response = session.get(URL,params=params,timeout=45)
                    response.raise_for_status()
                    obj = response.json()
                    if not isinstance(obj,dict) or 'articles' not in obj:
                        raise ValueError('No articles field; do not interpret a service error as zero news.')
                    payload = response.content
                    meta.update(url=response.url,retrieved_utc=datetime.now(timezone.utc).isoformat(),
                                sha256=hashlib.sha256(payload).hexdigest(),status=200)
                    path.write_bytes(payload)
                    meta_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
                    break
                except (requests.RequestException,ValueError) as exc:
                    meta['errors'].append({'attempt':attempt+1,'error':str(exc),
                                           'at_utc':datetime.now(timezone.utc).isoformat()})
                    if attempt<3:time.sleep(10*(attempt+1))
        if payload is None:
            meta.update(status='failed',n_returned=None,capped=None)
            meta_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
        else:
            articles = json.loads(payload)['articles']
            meta.update(n_returned=len(articles),capped=len(articles)>=250)
            for item in articles:
                records.append(dict(item,query_date=date,provider='GDELT DOC 2.0',
                                    retrieval_utc=meta['retrieved_utc']))
        audit.append(meta)
        pd.DataFrame(audit).drop(columns=['errors'],errors='ignore').to_csv(ROOT/'outputs/gdelt_daily_coverage.csv',index=False)
        if records:pd.DataFrame(records).to_csv(ROOT/'data/processed/gdelt_headlines.csv',index=False)
        print(date,meta['status'],meta.get('n_returned'),'capped',meta.get('capped'),flush=True)
    summary = {'source':URL,'query':QUERY,'start':START,'end_exclusive':END,
               'requested_days':len(audit),'successful_days':sum(x['status']==200 for x in audit),
               'failed_days':sum(x['status']!=200 for x in audit),
               'capped_days':sum(x.get('capped') is True for x in audit),
               'returned_records':len(records),'unique_urls':len({x.get('url') for x in records}),
               'text_scope':'Retrieved headlines; article bodies are not returned by this API.',
               'timestamp':'seendate is GDELT first-seen time, not verified first publication.',
               'zero_definition':'A failed or capped request cannot establish absence of news.',
               'coverage':'Daily relevance-ranked lists capped at 250; not a complete global news census.'}
    (ROOT/'outputs/gdelt_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return pd.DataFrame(records),pd.DataFrame(audit)


if __name__=='__main__':collect()
