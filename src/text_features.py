"""News attention and vocabulary about physical conflict and economic conditions.

Official statements and political assessments are not assigned sentiment scores.
Numerical lexical measures use only unquoted physical-context sentences that
pass conservative subject, attribution, policy and anaphora exclusions. Article
relevance and attention counts use the complete original text independently.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from .study_config import CONTENT_END_EXCLUSIVE
import hashlib
import json
import re
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
LM_URL = 'https://drive.usercontent.google.com/download?id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF&export=download&confirm=t'
LM_SHA = 'e2d1328682bab7d2187684fb9f5420bb730401c9eefc00daf835edd203f4859d'
TOKEN = re.compile(r'[A-Za-z]{2,}')
CONFLICT = re.compile(r'\b(?:war|wars|military|missiles?|drones?|bomb\w*|airstrikes?|strikes?|attacks?|hostilit\w*|ceasefire|blockad\w*|hormuz|naval|troops?|nuclear|uranium|shipping|tankers?|oil|refiner\w*|pipeline\w*)\b', re.I)
PHYSICAL = re.compile(r'\b(?:missiles?|drones?|bomb\w*|airstrikes?|strikes?|attacks?|blockad\w*|hormuz|naval|troops?|nuclear|uranium|shipping|tankers?|oil|refiner\w*|pipeline\w*|killed|injur\w*|casualt\w*|deaths?|displac\w*|destroy\w*|damag\w*)\b', re.I)
IRAN = re.compile(r'\b(?:iran\w*|tehran|hormuz|persian gulf|kharg)\b', re.I)
THREAT = re.compile(r'\b(?:risk\w*|threat\w*|fear\w*|warn\w*|could|might|possible|potential|uncertain\w*)\b', re.I)
ACT = re.compile(r'\b(?:attacked|struck|killed|bombed|launched|destroyed|damaged|injured|closed|blocked|suspended|fired|sank)\b', re.I)
DIPLOMACY = re.compile(r'\b(?:ceasefire|talks|negotiat\w*|truce|peace|agreement|diplomac\w*)\b', re.I)
POLITICAL = re.compile(
    r'\b(?:trump|biden|hegseth|netanyahu|khamenei|pezeshkian|araghchi|vance|rubio|'
    r'ghalibaf|qalibaf|witkoff|kushner|lutnick|bessent|caine|huckabee|albusaidi|'
    r'katz|starmer|macron|merz|zelensky\w*|putin|lavrov|harris|pelosi|schumer|'
    r'white\s+house|pentagon|centcom|irgc|idf|revolutionary\s+guards?|'
    r'president\w*|minist(?:er|ers|ry|ries)|governments?|administrations?|'
    r'congress\w*|senat\w*|parliament\w*|cabinet|officials?|authorities|'
    r'negotiators?|envoys?|spokes(?:person|people|man|men|woman|women)|'
    r'secretar(?:y|ies)|leaders?|generals?|admirals?|commanders?|'
    r'ambassadors?|diplomats?|governors?|chancellors?|monarch\w*|commissioners?|'
    r'crown\s+prince|supreme\s+leader|politic\w*|policy|policies|'
    r'sanction\w*|tariff\w*|deals?|agreements?|accords?|'
    r'negotiat\w*|diploma\w*|ceasefire\w*|truce\w*|peace\w*|'
    r'election\w*|republican\w*|democrat\w*|conservatives?|reformers?|'
    r'hardliners?|nationalism|regimes?|ultimatum\w*|'
    r'reparations?|concessions?|demands?|legitim\w*|illegal|unlawful|'
    r'justif\w*|victor\w*|defeat\w*|strateg\w*|objectives?|goals?|aims?|'
    r'laws?|legal\w*|prosecut\w*|statutes?|tax\w*|protocols?|legislat\w*|'
    r'missions?|operations?|assassinat\w*|achiev\w*|military\s+thinking|'
    r'plans?|planning|propos\w*|decisions?|options?|agendas?|scenarios?|'
    r'support\w*|oppos\w*|retaliat\w*|escalat\w*|geopolitic\w*|'
    r'iaea|atomic\s+energy\s+agency|united\s+nations|eu|icc|icj|imo|nato|hamas|hezbollah|houthis?)\b', re.I)
# Exclude complete sentences containing reported speech, even when no speaker
# name survives quotation removal. Pronouns can refer to a speaker or office in
# the preceding sentence; accepting that ambiguity would defeat the exclusion.
ATTRIBUTION = re.compile(
    r'\b(?:say|says|said|saying|tell|tells|told|stat(?:e|es|ed|ing)|'
    r'announc\w*|claim\w*|argu\w*|believ\w*|according\s+to|post(?:s|ed|ing)?|'
    r'vow\w*|warn\w*|threaten\w*|insist\w*|declar\w*|confirm\w*|'
    r'report(?:s|ed|ing)?|assert\w*|alleg\w*|acknowledg\w*|'
    r'pledge\w*|promis\w*|den(?:y|ies|ied)|suggest\w*|predict\w*|estimat\w*|'
    r'comment\w*|describ\w*|explain\w*|remark\w*|respond\w*|'
    r'accus\w*|critic\w*|defend\w*|blam\w*|prais\w*|dismiss\w*|'
    r'reject\w*|call(?:s|ed|ing)?)\b', re.I)
ANAPHORA = re.compile(r'\b(?:he|him|his|she|her|hers|we|us|our|ours|i|me|my|mine|they|them|their|theirs|it|its)\b', re.I)


def lexicons():
    path = ROOT / 'data/raw/lexicons/LoughranMcDonald_MasterDictionary.csv'
    if not path.exists():
        r = requests.get(LM_URL, timeout=90)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != LM_SHA:
        raise ValueError('LM source differs from the verified March 2026 release.')
    master = pd.read_csv(path, low_memory=False)
    lists = {c.lower(): set(master.loc[pd.to_numeric(master[c], errors='coerce').gt(0), 'Word'].str.upper())
             for c in ['Negative', 'Positive', 'Uncertainty']}
    (path.parent/'manifest.json').write_text(json.dumps({'url':LM_URL,'release':'March 2026','sha256':LM_SHA,
        'checked_utc':datetime.now(timezone.utc).isoformat(),'sizes':{k:len(v) for k,v in lists.items()}},indent=2))
    return lists


def sentence_features(title, body):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', body) if s.strip()]
    context = [s for s in sentences if CONFLICT.search(s)]
    # Remove whole quotation spans BEFORE splitting: an interior sentence in a
    # multi-sentence quote may have no quotation mark of its own.
    # Reject the surrounding quotation-bearing sentence too. Preserve a final
    # stop inside a quotation so a following independent sentence stays separate.
    def quote_marker(match):
        stop = '. ' if re.search(r'[.!?][\"”’]$', match.group()) else ' '
        subject = ' EXCLUDED_SUBJECT' if POLITICAL.search(match.group()) or ATTRIBUTION.search(match.group()) else ''
        marker = ' EXCLUDED_QUOTATION' + subject
        return marker + ('\n' + marker) * match.group().count('\n') + stop
    # Strip spans globally first, preserving paragraph boundaries, so removing
    # an attributed first paragraph cannot expose the middle of a long quote.
    unquoted = re.sub(r'“[^”]*”|"[^"]*"|‘[^’]*’', quote_marker, body, flags=re.S)
    unquoted = re.sub(r'[“"][^”"]*$', quote_marker, unquoted, flags=re.S)
    # Paragraph context catches an implicit viewpoint in one sentence whose
    # speaker or policy subject appears elsewhere in the same paragraph.
    physical_paragraphs = [p for p in re.split(r'\n+', unquoted)
                           if not POLITICAL.search(p) and not ATTRIBUTION.search(p)
                           and 'EXCLUDED_SUBJECT' not in p]
    unquoted = '\n'.join(physical_paragraphs)
    unquoted_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', unquoted) if s.strip()]
    physical = [s for s in unquoted_sentences if CONFLICT.search(s) and PHYSICAL.search(s)
                and '?' not in s and 'EXCLUDED_QUOTATION' not in s
                and not any(rule.search(s) for rule in (POLITICAL, ATTRIBUTION, ANAPHORA))]
    counts = Counter(TOKEN.findall(' '.join(physical).upper()))
    # Classification is literal subject retrieval, not a probability of war.
    relevant = bool(CONFLICT.search(title) and IRAN.search(title)) or sum(bool(IRAN.search(s)) for s in context) >= 2
    return {'war_relevant': relevant, 'n_sentences':len(sentences), 'n_context_sentences':len(context),
            'n_physical_sentences':len(physical), 'threat_mentions':sum(bool(THREAT.search(s)) for s in physical),
            'act_mentions':sum(bool(ACT.search(s)) for s in physical),
            'diplomacy_mentions':sum(bool(DIPLOMACY.search(s)) for s in physical),
            'physical_text':' '.join(physical), 'n_words':sum(counts.values()), 'n_distinct':len(counts)}, counts


def score_counts(counters, lists):
    """Exact Assignment 1 Eq.1 weights; IDF fits the supplied corpus only."""
    vocabulary = set().union(*lists.values())
    df = Counter()
    for c in counters:
        df.update(set(c).intersection(vocabulary))
    n = len(counters)
    output = []
    for c in counters:
        total = sum(c.values())
        row = {}
        for category, words in lists.items():
            hits = sum(c[w] for w in words)
            row[category+'_count'] = hits
            row[category+'_pct'] = 100*hits/total if total else np.nan
            row[category+'_tfidf'] = sum((1+np.log(c[w]))/(1+np.log(total/len(c))) * np.log(n/df[w])
                for w in words.intersection(c)) if total else np.nan
        output.append(row)
    return pd.DataFrame(output)


def map_times_to_sessions(times, sessions, cutoff_hour=15):
    """Next observed US equity session with 15:00 ET at/after the timestamp."""
    sessions = pd.DatetimeIndex(sessions).normalize().sort_values().unique()
    cutoffs = (sessions + pd.Timedelta(hours=cutoff_hour)).tz_localize('America/New_York').tz_convert('UTC')
    stamps = pd.to_datetime(times, utc=True)
    positions = cutoffs.searchsorted(stamps, side='left')
    return pd.Series([sessions[i] if i < len(sessions) else pd.NaT for i in positions], index=times.index)


def build():
    (ROOT/'outputs').mkdir(parents=True, exist_ok=True)
    docs = pd.read_csv(ROOT/'data/processed/news_documents.csv').fillna('')
    docs = docs.loc[docs.exclusion.eq('')].copy().reset_index(drop=True)
    original_count = len(docs)
    pub = pd.to_datetime(docs.published_utc, utc=True)
    mod = pd.to_datetime(docs.modified_utc, utc=True, errors='coerce').fillna(pub)
    late = pd.concat([pub,mod],axis=1).max(axis=1).ge(pd.Timestamp(CONTENT_END_EXCLUSIVE,tz='America/New_York').tz_convert('UTC'))
    late_count = int(late.sum())
    # A later revised body cannot enter either timing convention or IDF fitting.
    docs = docs.loc[~late].reset_index(drop=True)
    records, counters = [], []
    for r in docs.itertuples():
        feat, c = sentence_features(r.title or r.archive_title, r.body)
        records.append(feat); counters.append(c)
    frame = pd.concat([docs.drop(columns=['body']), pd.DataFrame(records), score_counts(counters,lexicons())],axis=1)
    publication = pd.to_datetime(frame.published_utc, utc=True)
    modification = pd.to_datetime(frame.modified_utc, utc=True, errors='coerce').fillna(publication)
    # Later modifications may change the entire body; use the later clock conservatively.
    effective = pd.concat([publication, modification],axis=1).max(axis=1)
    frame['effective_utc'] = effective
    frame['modified_after_cutoff'] = effective.ge(pd.Timestamp(CONTENT_END_EXCLUSIVE,tz='America/New_York').tz_convert('UTC'))
    changes = pd.read_csv(ROOT/'data/processed/market_changes.csv',index_col=0,parse_dates=True)
    all_sessions = changes.dropna(subset=['sp500']).index
    sessions = all_sessions[all_sessions >= '2026-01-01']
    frame['event_date'] = map_times_to_sessions(effective,all_sessions)
    frame['publication_event_date'] = map_times_to_sessions(publication,all_sessions)
    frame.to_csv(ROOT/'data/processed/news_features.csv',index=False)
    daily = pd.DataFrame(index=sessions)
    for clock, suffix in [('event_date',''), ('publication_event_date','_publication')]:
        sample = frame.loc[frame[clock].notna() & ~frame.modified_after_cutoff].copy()
        sample['war_n'] = sample.war_relevant.astype(int)
        grp = sample.groupby(clock)
        daily['articles'+suffix] = grp.size().reindex(sessions,fill_value=0)
        daily['war_articles'+suffix] = grp.war_n.sum().reindex(sessions,fill_value=0)
        war = sample.loc[sample.war_relevant & sample.n_words.ge(20)]
        g = war.groupby(clock)
        for c in ['negative','positive','uncertainty']:
            daily[c+'_pct'+suffix] = 100*g[c+'_count'].sum()/g.n_words.sum()
            daily[c+'_tfidf'+suffix] = g[c+'_tfidf'].mean()
        for c in ['threat','act','diplomacy']:
            daily[c+'_pct'+suffix] = 100*g[c+'_mentions'].sum()/g.n_physical_sentences.sum()
    previous = pd.Series(all_sessions,index=all_sessions).shift(1).reindex(sessions)
    daily['interval_days'] = (pd.Series(sessions,index=sessions)-previous).dt.days
    daily['war_attention'] = daily.war_articles/daily.interval_days
    daily['war_share_pct'] = 100*daily.war_articles/daily.articles.replace(0,np.nan)
    daily['log_attention'] = np.log1p(daily.war_attention)
    daily['d_log_attention'] = daily.log_attention.diff()
    daily['d_negative_pct'] = daily.negative_pct.diff()
    daily['d_uncertainty_pct'] = daily.uncertainty_pct.diff()
    daily['d_threat_pct'] = daily.threat_pct.diff()
    daily.index.name = 'date'
    daily.to_csv(ROOT/'data/processed/news_daily.csv')
    terms=Counter()
    for c,war in zip(counters,frame.war_relevant):
        if war:terms.update(c)
    pd.DataFrame(terms.most_common(60),columns=['word','count']).to_csv(ROOT/'outputs/common_words.csv',index=False)
    audit = {'eligible_news_documents':original_count,'within_cutoff_documents':len(frame),'war_relevant_documents':int(frame.war_relevant.sum()),
             'numeric_text_eligible':int((frame.war_relevant & frame.n_words.ge(20)).sum()),
             'modified_after_cutoff':late_count,'mapped_effective_date':int(frame.event_date.notna().sum()),
             'scoring_target':'Unquoted physical conflict/economic context after paragraph-level official-subject, policy and attribution exclusions, plus sentence-level personal-anaphora exclusions. Attention counts independently retain all relevant articles.',
             'limits':'Conservative heuristic exclusions can omit factual reporting and cannot guarantee semantic separation; no independent labeled accuracy; financial LM vocabulary transferred to news; current revised text. Absence of eligible text yields a missing value, not neutrality.',
             'idf':'Full eligible news corpus, retrospective; no prediction claim.'}
    (ROOT/'outputs/text_diagnostics.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
    return frame,daily


if __name__ == '__main__':
    build()
