"""Audit every retrieved social candidate without sentiment scores or ratings.

This is an automated data/attribution audit with selected analyst context notes,
not independent human labeling, a semantic relevance classifier, or a census of
all public communication. Existing source and candidate files are never changed.
"""
from pathlib import Path
import hashlib
import html
import json
import re
import warnings

import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

ROOT = Path(__file__).resolve().parents[1]
IRAN = re.compile(r"\b(?:iran(?:ian|ians)?|hormuz|tehran|khamenei|kharg|epic fury|midnight hammer)\b", re.I)
URL = re.compile(r"https?://[^\s<>]+", re.I)
RT = re.compile(r"^\s*\[?RT(?::|\s*@)", re.I)
ATTRIBUTION = re.compile(r"^\s*(?:From\s+[^:]{1,90}:|FoxNews:|OFFICIAL STATEMENT OF IRAN:)", re.I)
CONTEXT_NOTES = {
    '116069904214662301': 'Television-host/dinner discussion; Midnight Hammer appears in a list of claimed achievements, not a new Iran announcement.',
    '116199640636892669': 'Iranian women football players and an asylum request; humanitarian/sports context.',
    '116200028617921781': 'Iranian women football players and discussion with Australia; humanitarian/sports context.',
    '116216801278101254': 'Iran national football team and World Cup safety; humanitarian/sports context.',
    '116290846597255331': 'FISA Section 702 extension argument invokes Iran operations; mixed legislative context.',
    '116346686438015338': 'Employment and trade statistics plus an Iran clause; multiple attitude targets.',
    '116366072136989268': 'Caption introducing an Iranian statement; attached media content is not transcribed in this text field.',
    '116388414661607621': 'Explicitly introduced as material from Newt Gingrich; not solely words authored by the account holder.',
    '116404390855989138': 'FISA Section 702 extension argument invokes Iran operations; mixed legislative context.',
    '116454250724920799': 'Brief account commentary precedes a named Marc Thiessen headline/link; mixed author voices.',
    '116575104401917058': 'China/US-economic-performance discussion with an Iran reference in a broader list.',
    '116602192066577324': 'Senate endorsement with an Iran/nuclear clause; political campaign content, not a standalone conflict update.',
    '116746683568766400': 'Senate endorsement with Iran, nuclear, Hormuz and gasoline claims embedded in the campaign text.',
    '116754019906128212': 'Revised Senate endorsement with Iran/Hormuz/gasoline material; differs from the prior record.',
    '116825821568797363': 'Book/media criticism with an Iran sentence appended; mixed subject.',
    '116832995541119171': 'Polling claim and Iran nuclear clause; does not itself measure public opinion on the war.',
    '116841220861327348': 'Midterm convention announcement with oil/Iran language embedded; mixed campaign/economic text.',
    '116868292534134605': 'Explicit quotation attributed to Shou Chew about TikTok views; not independent war-sentiment evidence.',
    '116918800567887958': 'A 2026 link share referring to a clip from 1980; publication time is not the historical statement time.',
    '116969149016233329': 'Saudi civilian nuclear agreement; Iran is a comparator, not the main agreement discussed.',
    '117031820986118471': 'Polling/election statement with Iran among claimed achievements; multiple attitude targets.',
    '117039401914044983': 'Repeated polling/election statement with Iran among claimed achievements.',
    '117049594693768642': 'Personnel/media discussion with Iran operations as one supporting example.',
    '117107001767972921': 'Television/media and polling criticism with Iran in a broader claimed-achievement list.',
    '117107215379637520': 'Korean military-exercise statement with an explicitly described aside concerning Iran.',
    '117108149935654446': 'Marked repost of a Korean military-exercise statement with an Iran aside.',
    '117151357800143501': 'Polling/election discussion with an Iran clause; multiple attitude targets.',
    '117151563892688595': 'Marked repost of polling/election discussion with an Iran clause.',
    '117156395248393739': 'Revised Korean military-exercise statement with an Iran aside.',
    '117174459387909497': 'News headline about a domestic enforcement allegation mentioning Iran/IRGC; not a direct military update.',
    '117185823993660126': 'Automobile/tariff/Canada statement with a brief Iran comparison.',
    '117206830744287806': 'Brief account commentary joined to a Syria/Hormuz headline and link; mixed author voices.',
    '117215105900362553': 'Podcast promotion lists Iran among topics; it is not a transcript of the discussion.',
    '117270009579288120': 'Diesel-price attribution to the Russia/Ukraine war contrasted with Iran; negation and target matter.',
}


def decode_text(value):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', MarkupResemblesLocatorWarning)
        return html.unescape(BeautifulSoup(value or '', 'html.parser').get_text(' ', strip=True))


def main():
    raw_path = ROOT / 'data/raw/social/truth_archive.json'
    candidates_path = ROOT / 'data/processed/social_candidates.csv'
    raw_bytes = raw_path.read_bytes()
    archive = json.loads(raw_bytes)
    source = {str(row['id']): row for row in archive}
    data = pd.read_csv(candidates_path, dtype=str).fillna('')
    if data.id.duplicated().any():
        raise ValueError('Candidate IDs are not unique.')

    # Independently audit every archived record in the date window and rederive
    # candidate membership. A set comparison detects both missing and extra rows.
    expected = set()
    empty_with_media = 0
    for record in archive:
        stamp = pd.Timestamp(record['created_at']).tz_convert('America/New_York')
        if pd.Timestamp('2026-01-01', tz='America/New_York') <= stamp < pd.Timestamp('2026-09-19', tz='America/New_York'):
            decoded = decode_text(record.get('content', ''))
            if not decoded and record.get('media'):
                empty_with_media += 1
            if IRAN.search(decoded):
                expected.add(str(record['id']))
    if set(data.id) != expected:
        raise ValueError('Candidate membership differs from independent snapshot audit.')

    rows = []
    for row in data.to_dict('records'):
        original = source[row['id']]
        text = row['text']
        if decode_text(original.get('content', '')) != text:
            raise ValueError(f"Text mismatch for {row['id']}")
        stamp = pd.Timestamp(original['created_at']).tz_convert('America/New_York')
        if str(stamp.date()) != row['date_et']:
            raise ValueError(f"Date mismatch for {row['id']}")
        urls = URL.findall(text)
        body = URL.sub('', text).strip()
        body_matches = sorted(set(match.group().lower() for match in IRAN.finditer(body)))
        url_matches = sorted(set(match.group().lower() for url in urls for match in IRAN.finditer(url)))
        marked_rt = bool(RT.search(text))
        attributed = bool(ATTRIBUTION.search(text))
        if marked_rt:
            form = 'Explicit RT marker'
        elif not body and urls:
            form = 'URL only'
        elif attributed:
            form = 'Explicit attribution prefix'
        elif urls:
            form = 'Text plus URL'
        else:
            form = 'Text without URL or explicit attribution prefix'
        location = 'Body and URL' if body_matches and url_matches else ('Body' if body_matches else 'URL only')
        if marked_rt:
            authorship = 'Repost cue observed; verify original source before attributing quoted words'
        elif attributed:
            authorship = 'Named source or document attribution observed; multiple voices possible'
        elif urls:
            authorship = 'Link share; headline/source text and account commentary not separated by schema'
        else:
            authorship = 'No explicit repost/quotation prefix; original authorship not independently verified'
        # No rules infer the emotion, attitude, credibility or merit of the author.
        rows.append({**row,
            'source_form': form,
            'relevance_match_location': location,
            'body_matched_keywords': '|'.join(body_matches),
            'url_matched_keywords': '|'.join(url_matches),
            'explicit_repost_marker': marked_rt,
            'explicit_attribution_prefix': attributed,
            'contains_url': bool(urls),
            'url_only_record': bool(urls) and not bool(body),
            'contains_double_quote_character': bool(re.search(r'[\"“”]', text)),
            'authorship_audit': authorship,
            'media_count': len(original.get('media', [])),
            'media_text_not_reviewed': bool(original.get('media')),
            'qualitative_context_note': CONTEXT_NOTES.get(row['id'], ''),
            'audit_method': 'Literal rules over every full text; selected analyst context notes; no human gold labels',
        })
    audit = pd.DataFrame(rows)
    outputs = ROOT / 'outputs'
    outputs.mkdir(exist_ok=True)
    audit.to_csv(outputs / 'social_relevance_audit.csv', index=False)
    summary = {
        'candidate_rows_audited': len(audit),
        'candidate_ids_match_independent_raw_retrieval': True,
        'all_text_and_local_dates_match_raw_snapshot': True,
        'candidate_csv_sha256': hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
        'archive_sha256': hashlib.sha256(raw_bytes).hexdigest(),
        'source_form_counts': {k: int(v) for k, v in audit.source_form.value_counts().items()},
        'relevance_match_location_counts': {k: int(v) for k, v in audit.relevance_match_location.value_counts().items()},
        'explicit_repost_marker_records': int(audit.explicit_repost_marker.sum()),
        'explicit_self_repost_prefix_records': int(audit.text.str.match(r'^\s*RT\s+@realDonaldTrump', case=False).sum()),
        'explicit_attribution_prefix_records': int(audit.explicit_attribution_prefix.sum()),
        'url_only_records': int(audit.url_only_record.sum()),
        'records_with_urls': int(audit.contains_url.sum()),
        'records_with_unreviewed_media': int(audit.media_text_not_reviewed.sum()),
        'empty_text_records_with_media_in_full_2026_window': empty_with_media,
        'candidate_records_with_identical_text_other_id': int(audit.text.duplicated(keep=False).sum()),
        'unique_candidate_texts': int(audit.text.nunique()),
        'analyst_context_notes': int(audit.qualitative_context_note.ne('').sum()),
        'sentiment_scores_or_rankings_produced': False,
        'interpretation': 'Counts describe observable archive fields and literal cues; no verified complete corpus, authorship proof, human gold labels, sentiment labels or risk ratings.',
    }
    (outputs / 'social_relevance_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
