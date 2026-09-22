"""Collect a dated, auditable Iran-tag news corpus without an API credential.

Archive pages are a discovery source, not an exhaustive census of global news.
Full article bodies are retained locally only. Live blogs are excluded because
their final text incorporates updates after the initial publication timestamp.
"""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from .study_config import CONTEXT_START, CONTENT_END_EXCLUSIVE, MARKET_END, NEWS_DISCOVERY_SNAPSHOT

ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp(CONTEXT_START, tz='UTC')
END = pd.Timestamp(CONTENT_END_EXCLUSIVE, tz='America/New_York').tz_convert('UTC')
RAW = ROOT / 'data/raw/news'
MONTHS = {m: i+1 for i, m in enumerate('jan feb mar apr may jun jul aug sep oct nov dec'.split())}


def fetch(url, path):
    if path.exists():
        return path.read_bytes()
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=40, headers={'User-Agent': 'Mozilla/5.0 (academic text research)'})
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            return response.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 + 2*attempt)


def parse_archive(payload, page):
    soup = BeautifulSoup(payload, 'html.parser')
    records = []
    for a in soup.select('a[aria-label][href]'):
        href = a['href'].split('#')[0].split('?')[0]
        match = re.search(r'/(20\d{2})/([a-z]{3})/(\d{2})/', href)
        if not match or not href.startswith('/'):
            continue
        year, month, day = match.groups()
        date = f'{year}-{MONTHS[month]:02d}-{day}'
        records.append({'url': 'https://www.theguardian.com'+href, 'archive_title': a['aria-label'].strip(),
                        'url_date': date, 'archive_page': page, 'section': href.split('/')[1]})
    return records


def discover():
    records = []
    page_meta = []
    # Each page contains 20 cards. Continue until every date on a page predates 2026.
    for page in range(1, 301):
        url = f'https://www.theguardian.com/world/iran?page={page}'
        path = RAW / ('archive_'+NEWS_DISCOVERY_SNAPSHOT) / f'{page:03d}.html'
        payload = fetch(url, path)
        rows = parse_archive(payload, page)
        if not rows:
            raise ValueError(f'No dated cards on archive page {page}; do not treat parser failure as an empty day.')
        records.extend(rows)
        page_meta.append({'page': page, 'url': url, 'sha256': hashlib.sha256(payload).hexdigest(),
                          'earliest': min(r['url_date'] for r in rows), 'latest': max(r['url_date'] for r in rows)})
        if page % 10 == 0:
            print('Archive', page, page_meta[-1]['earliest'], flush=True)
        if max(r['url_date'] for r in rows) < '2026-01-01':
            break
        time.sleep(.15)
    else:
        raise RuntimeError('Archive cap reached before the start date.')
    frame = pd.DataFrame(records).drop_duplicates('url')
    frame = frame.loc[frame.url_date.between(CONTEXT_START, MARKET_END)].copy()
    frame['exclusion'] = ''
    frame.loc[~frame.section.isin(['world', 'us-news', 'business', 'environment', 'global-development', 'science']), 'exclusion'] = 'non-news section'
    frame.loc[frame.url.str.contains('/live/|/video/|/audio/|/gallery/'), 'exclusion'] = 'live blog or multimedia'
    # Exclude explicit opinion and explainers only if their actual section says so;
    # ordinary analytical articles in news sections remain in the sampling frame.
    return frame, page_meta


def parse_article(payload):
    soup = BeautifulSoup(payload, 'html.parser')
    metadata = {}
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            obj = json.loads(script.string or script.get_text())
            objects = obj if isinstance(obj, list) else [obj]
            for o in objects:
                if isinstance(o, dict) and o.get('@type') in ('NewsArticle', 'Article', 'ReportageNewsArticle', 'AnalysisNewsArticle'):
                    metadata = o
                    break
        except (ValueError, TypeError):
            continue
    paragraphs = soup.select('[data-gu-name="body"] p') or soup.select('div.article-body-commercial-selector p')
    body = '\n'.join(p.get_text(' ', strip=True) for p in paragraphs)
    if not body:
        body = metadata.get('articleBody', '')
    return {'title': metadata.get('headline', ''), 'published_utc': metadata.get('datePublished', ''),
            'modified_utc': metadata.get('dateModified', ''), 'body': body,
            'body_words': len(re.findall(r'[A-Za-z]+', body))}


def article(row):
    if row['exclusion']:
        return row
    key = hashlib.sha256(row['url'].encode()).hexdigest()[:20]
    path = RAW / 'articles' / (key+'.html')
    try:
        payload = fetch(row['url'], path)
        row.update(parse_article(payload))
        row['raw_path'] = path.relative_to(ROOT).as_posix()
        row['sha256'] = hashlib.sha256(payload).hexdigest()
        if row['body_words'] < 80:
            row['exclusion'] = 'fewer than 80 extracted body words'
        elif not row['published_utc']:
            row['exclusion'] = 'missing publication timestamp'
        else:
            stamp = pd.Timestamp(row['published_utc'])
            if stamp.tzinfo is None:
                row['exclusion'] = 'publication timezone missing'
            elif not START <= stamp.tz_convert('UTC') < END:
                row['exclusion'] = 'timestamp outside information set'
        time.sleep(.1)
    except Exception as exc:
        row['exclusion'] = 'download/parse failure: '+str(exc)[:160]
    return row


def collect():
    RAW.mkdir(parents=True, exist_ok=True)
    discovered, pages = discover()
    discovered.to_csv(RAW / 'discovery.csv', index=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = []
        for i, row in enumerate(pool.map(article, discovered.to_dict('records')), 1):
            rows.append(row)
            if i % 100 == 0:
                print('Articles', i, '/', len(discovered), flush=True)
    frame = pd.DataFrame(rows)
    # Exact duplicated bodies are retained in audit but removed from the corpus.
    frame['body_hash'] = frame.body.fillna('').map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    eligible = frame.exclusion.eq('')
    duplicate = frame.loc[eligible].duplicated('body_hash', keep='first')
    frame.loc[duplicate[duplicate].index, 'exclusion'] = 'exact duplicate body'
    (ROOT / 'data/processed').mkdir(parents=True, exist_ok=True)
    frame.to_csv(ROOT / 'data/processed/news_documents.csv', index=False)
    metadata = {'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'start_inclusive': str(START),
                'end_exclusive': str(END), 'source': 'Guardian Iran tag archive; single-outlet sample',
                'archive_pages': pages, 'discovered_unique_2026_urls': len(frame),
                'included': int(frame.exclusion.eq('').sum()), 'exclusions': frame.exclusion.value_counts().to_dict(),
                'content_vintage': 'Bodies as retrieved, potentially edited since original publication; not archived real-time text.',
                'coverage': 'All discovered eligible cards through complete archive pagination; no claim of all global news.'}
    (RAW / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in metadata.items() if k != 'archive_pages'}, indent=2))
    return frame


if __name__ == '__main__':
    collect()
