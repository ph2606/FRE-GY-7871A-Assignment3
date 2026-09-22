"""Free GDELT daily event archive; retain source fields for Iran-related records.

The bulk archive avoids the DOC API result cap. It is an automatically extracted
event database, not full article text or a complete news census. Goldstein and
tone fields are not used. Daily file availability is later than its news date.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib, io, json, time, zipfile
import numpy as np
import pandas as pd
import requests
from .study_config import CONTEXT_START, MARKET_END

ROOT=Path(__file__).resolve().parents[1]
FIELDS={0:'event_id',1:'event_date',7:'actor1_country',17:'actor2_country',
        26:'event_code',28:'event_root',37:'actor1_geo_country',44:'actor2_geo_country',
        51:'action_country',56:'date_added',57:'url'}
SOURCE='https://data.gdeltproject.org/events/'


def select_records(payload, date):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names=archive.namelist()
        if len(names)!=1:raise ValueError('Unexpected archive members')
        with archive.open(names[0]) as stream:
            frame=pd.read_csv(stream,sep='\t',header=None,usecols=list(FIELDS),dtype=str,
                              keep_default_na=False,encoding='utf-8',encoding_errors='replace')
    frame=frame.rename(columns=FIELDS)
    relevant=frame.actor1_country.eq('IRN') | frame.actor2_country.eq('IRN') | frame.action_country.eq('IR')
    sample=frame.loc[relevant].copy()
    sample['archive_date']=date
    return sample,len(frame)


def one_day(day):
    date=day.strftime('%Y-%m-%d');stamp=day.strftime('%Y%m%d')
    folder=ROOT/'data/raw/gdelt/iran_events';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(date+'.csv');metadata_path=folder/(date+'.metadata.json')
    if path.exists() and metadata_path.exists():
        metadata=json.loads(metadata_path.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['subset_sha256']:
            raise ValueError('Cached Iran-event subset hash mismatch')
        return pd.read_csv(path,dtype=str,keep_default_na=False),metadata
    url=SOURCE+stamp+'.export.CSV.zip';errors=[]
    for attempt in range(3):
        try:
            prior=ROOT/'data/raw/gdelt/bulk'/f'{stamp}.export.CSV.zip'
            if prior.exists():payload=prior.read_bytes()
            else:
                response=requests.get(url,timeout=60);response.raise_for_status();payload=response.content
            frame,total=select_records(payload,date)
            frame.to_csv(path,index=False)
            metadata={'date':date,'url':url,'status':200,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                      'response_sha256':hashlib.sha256(payload).hexdigest(),'response_bytes':len(payload),
                      'subset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'global_event_records':total,'iran_related_records':len(frame),'errors':errors,
                      'retention':'Selected source fields for Iran actors/action location; full-response hash retained.'}
            metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
            return frame,metadata
        except (requests.RequestException,ValueError,zipfile.BadZipFile) as exc:
            errors.append(str(exc))
            if attempt<2:time.sleep(3*(attempt+1))
    return pd.DataFrame(),{'date':date,'url':url,'status':'failed','errors':errors}


def classify_records(frame):
    frame=frame.copy()
    event=pd.to_datetime(frame.event_date,format='%Y%m%d',errors='coerce')
    added=pd.to_datetime(frame.date_added.str[:8],format='%Y%m%d',errors='coerce')
    frame['event_age_days']=(added-event).dt.days
    frame['recent_event']=frame.event_age_days.between(0,2)
    # CAMEO is a nominal taxonomy. No Goldstein/tone/risk score is calculated.
    frame['cessation_code']=frame.event_code.str.startswith('087')
    frame['fighting_code']=frame.event_root.isin(['19','20'])
    frame['event_category']=np.select([frame.cessation_code,frame.fighting_code],
          ['Truce/cessation/withdrawal code','Fighting/violence code'],default='Other Iran-related event code')
    frame['war_event_record']=(frame.cessation_code|frame.fighting_code) & frame.recent_event
    # GDELT promises daily files by 06:00 EST the following morning. 07:00
    # New York time is conservative both in winter and during daylight saving.
    release=pd.to_datetime(frame.archive_date)+pd.Timedelta(days=1,hours=7)
    frame['conservative_available_et']=release.dt.tz_localize('America/New_York').astype(str)
    return frame


def collect(workers=3):
    (ROOT/'outputs').mkdir(parents=True,exist_ok=True)
    (ROOT/'data/processed').mkdir(parents=True,exist_ok=True)
    dates=pd.date_range(CONTEXT_START,MARKET_END)
    frames=[];audit=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(one_day,date):date for date in dates}
        for task in as_completed(jobs):
            frame,metadata=task.result();audit.append(metadata)
            if len(frame):frames.append(frame)
            if len(audit)%10==0:print('Bulk days',len(audit),'/',len(dates),'failures',sum(x['status']!=200 for x in audit),flush=True)
    result=classify_records(pd.concat(frames,ignore_index=True).drop_duplicates(['event_id','date_added']))
    result.to_csv(ROOT/'data/processed/gdelt_iran_events.csv',index=False)
    coverage=pd.DataFrame(audit).sort_values('date')
    coverage.to_csv(ROOT/'outputs/gdelt_bulk_coverage.csv',index=False)
    summary={'requested_days':len(dates),'successful_days':int(coverage.status.eq(200).sum()),
      'failed_days':int(coverage.status.ne(200).sum()),'source_records':len(result),
      'source_urls':int(result.url.nunique()),'recent_war_event_records':int(result.war_event_record.sum()),
      'recent_war_source_urls':int(result.loc[result.war_event_record,'url'].nunique()),
      'source':SOURCE,'event_codebook':'https://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf',
      'scope':'Iran actor country IRN or action country IR; English-source GDELT 1.0 event extraction.',
      'limits':'Event database, not all articles or full text. Codes can be wrong; 0871 includes declared ceasefires without verified cessation. No tone/Goldstein scores used.',
      'timing':'Daily release at conservative next-calendar-day 07:00 ET; no intraday publication time inferred.'}
    (ROOT/'outputs/gdelt_bulk_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return result,coverage


if __name__=='__main__':collect()
