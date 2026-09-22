"""Auditable social-post retrieval and neutral qualitative topic inventory.

No sentiment ratings, rankings or evaluative scores of officials or policies.
The rows are archive records, not proof of original authorship or completeness.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import html
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
URL='https://ix.cnn.io/data/truth-social/truth_archive.json'
RELEVANCE=re.compile(r'\b(?:iran(?:ian|ians)?|hormuz|tehran|khamenei|kharg|epic fury|midnight hammer)\b',re.I)
TOPICS={
    'Shipping and energy':r'\b(?:hormuz|ship\w*|tanker\w*|oil|gasoline|energy|strait|pipeline\w*|ports?)\b',
    'Negotiations and conditions':r'\b(?:talk\w*|negotiat\w*|deal|agreement|ceasefire|truce|peace|if|unless)\b',
    'Military operations':r'\b(?:military|attack\w*|strike\w*|bomb\w*|missile\w*|navy|naval|war|epic fury)\b',
    'Nuclear matters':r'\b(?:nuclear|uranium|enrich\w*)\b',
    'Trade and sanctions':r'\b(?:tariff\w*|sanction\w*|trade|trading)\b',
    'People and humanitarian matters':r'\b(?:people|protest\w*|civilian\w*|hostage\w*|prisoner\w*|humanitarian)\b',
}


def build():
    (ROOT/'data/processed').mkdir(parents=True,exist_ok=True)
    (ROOT/'outputs').mkdir(parents=True,exist_ok=True)
    raw=ROOT/'data/raw/social';raw.mkdir(parents=True,exist_ok=True)
    path=raw/'truth_archive.json'
    if not path.exists():
        r=requests.get(URL,timeout=90);r.raise_for_status();path.write_bytes(r.content)
        (raw/'truth_archive.provenance.json').write_text(json.dumps({'url':URL,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
             'last_modified':r.headers.get('Last-Modified'),'sha256':hashlib.sha256(r.content).hexdigest()},indent=2))
    records=json.loads(path.read_text(encoding='utf-8'))
    rows=[]
    for r in records:
        stamp=pd.Timestamp(r['created_at']).tz_convert('America/New_York')
        if not pd.Timestamp('2026-01-01',tz='America/New_York')<=stamp<pd.Timestamp('2026-09-19',tz='America/New_York'):
            continue
        text=html.unescape(BeautifulSoup(r.get('content',''),'html.parser').get_text(' ',strip=True))
        # IDs stay strings: they exceed IEEE 754's exact integer range.
        rows.append({'id':str(r['id']),'platform':'Truth Social','created_at_utc':r['created_at'],
                     'created_at_et':stamp.isoformat(),'date_et':str(stamp.date()),'original_url':r['url'],'text':text,
                     'has_media':bool(r.get('media')),'empty_text':not bool(text.strip()),
                     'matched_keywords':'|'.join(sorted(set(x.lower() for x in RELEVANCE.findall(text)))),
                     'author_repost_status':'not distinguishable in archive schema'})
    frame=pd.DataFrame(rows).sort_values('created_at_utc')
    assert not frame.id.duplicated().any(),'Duplicate archive IDs must be audited.'
    selected=frame.loc[frame.matched_keywords.ne('')].copy()
    selected['literal_subjects']=selected.text.map(lambda t:'; '.join(name for name,pattern in TOPICS.items() if re.search(pattern,t,re.I)) or 'Context review needed')
    selected['conditional_wording']=selected.text.str.contains(r'\b(?:if|unless|until|provided that)\b',case=False,regex=True)
    selected['duplicate_text_other_id']=selected.duplicated('text',keep=False)
    selected.to_csv(ROOT/'data/processed/social_candidates.csv',index=False)
    frame.to_csv(raw/'truth_2026_complete_days_text.csv',index=False)
    monthly=selected.groupby(selected.date_et.str[:7]).size().rename('candidate_records')
    monthly.to_csv(ROOT/'outputs/social_monthly.csv')
    audit={'archive_records_all_years':len(records),'records_in_window':len(frame),'unique_ids':frame.id.nunique(),
           'empty_text_records':int(frame.empty_text.sum()),'media_records':int(frame.has_media.sum()),
           'iran_keyword_candidates':len(selected),'duplicate_candidate_text_records':int(selected.duplicate_text_other_id.sum()),
           'window':'2026-01-01 through 2026-09-18 inclusive; America/New_York',
           'regex':RELEVANCE.pattern,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
           'monthly_candidate_counts':monthly.to_dict(),'sentiment_scores_produced':False,
           'coverage':'All literal keyword matches in retrieved text; not all Iran posts. Media, implicit references, deletions and unknown archive gaps remain.',
           'x_coverage':'No verified full-archive X corpus acquired; Truth Social is not labeled Twitter.',
           'topics':'Multi-label literal subject rules, not evaluations or sentiment classifications.'}
    (ROOT/'outputs/social_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
    return selected


if __name__=='__main__':build()
