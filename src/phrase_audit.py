"""Headline retrieval and phrase-scope audit, separate from sentiment analysis.

The conservative comparator is an explicitly defined audit baseline, not a
historical classifier or the current body's relevance rule. Expanded retrieval
finds candidates, not verified war events. Synthetic cases test software rules;
they do not supply empirical precision, recall, or predictive accuracy.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ENTITY = r'\biran(?:ian|ians)?\b'
LEGACY_CONFLICT = r'\b(?:wars?|warfare|conflicts?|fighting|military)\b'
EXPANDED_ENTITY = r'\b(?:iran(?:ian|ians)?|tehran|hormuz|kharg|persian\s+gulf|fordow|fordo|natanz|bushehr|isfahan|irgc)\b'
GROUPS = {
    'legacy_conflict': LEGACY_CONFLICT,
    'physical_action': r'\b(?:air[ -]?strikes?|strikes?|struck|attacks?|attacked|bomb\w*|missiles?|drones?|naval|troops?|casualties|killed|injured|hostilities)\b',
    'shipping_access': r'\b(?:hormuz|straits?|tankers?|shipping|ships?|vessels?|waterways?|choke[ -]?points?|blockad\w*|ports?|sea\s+mines?)\b',
    'energy_supply': r'\b(?:oil|gas|crude|refiner\w*|pipelines?|fuel|energy|lng)\b',
    'nuclear_facility': r'\b(?:nuclear|uranium|enrich\w*|centrifuges?|fordow|fordo|natanz|bushehr)\b',
    'cessation_diplomacy': r'\b(?:cease[ -]?fire|truce|peace|negotiat\w*|talks?|agreements?|deals?)\b',
    'operation_code': r'\b(?:epic\s+fury|midnight\s+hammer|project\s+freedom|rising\s+lion|roaring\s+lion)\b',
}
NEGATION = r"\b(?:not|no|never|neither|nor|without|cannot|can['’]t|won['’]t|don['’]t|doesn['’]t|didn['’]t|isn['’]t|aren['’]t|hasn['’]t|haven['’]t|den(?:y|ies|ied))\b"
CONDITIONAL = r'\b(?:if|unless|until|provided\s+that|assuming|subject\s+to|could|may|might|would|possible|potential|expected\s+to)\b'
ATTRIBUTION = r'\b(?:says?|said|claims?|claimed|warns?|warned|vows?|vowed|announces?|announced|denies|denied|according\s+to|insists?|insisted|reports?|reported)\b'
QUOTATION = r'“[^”]+”|‘[^’]+’|"[^"]+"|(?<!\w)\x27[^\x27]+\x27(?!\w)'
LABOR = r'\b(?:workers?|unions?|nurses?|teachers?|wages?|walkout|industrial\s+action|picket\w*)\b'
TRADE_WAR = r'\b(?:trade|tariff|price|culture)\s+war\b'
AMBIGUOUS_TERMS = r'\b(?:strikes?|gas|deal|end|over|extend\w*|extension|operation|freedom|midnight)\b'
TOKEN = re.compile(r"cease[ -]?fire|[A-Za-z]+(?:['’][A-Za-z]+)?", re.I)
CLAUSE_SPLIT = re.compile(r'[,.!?;:…–—]|\b(?:but|while|whereas|if|unless|until)\b', re.I)
TARGETS = {'war': 'war', 'wars': 'war', 'conflict': 'war', 'conflicts': 'war', 'hostilities': 'war',
           'ceasefire': 'ceasefire', 'truce': 'ceasefire', 'deadline': 'deadline', 'deadlines': 'deadline',
           'talks': 'talks', 'negotiations': 'talks', 'discussion': 'talks', 'discussions': 'talks',
           'deal': 'agreement', 'agreement': 'agreement'}
ACTION_PATTERNS = {
    'extension': re.compile(r'extend(?:s|ed|ing)?|extensions?|prolong(?:s|ed|ing|ation)?|renew(?:s|ed|ing|al)?', re.I),
    'continuation': re.compile(r'continu(?:e|es|ed|ing|ation)|drags?', re.I),
    'collapse_expiry': re.compile(r'collaps(?:e|es|ed|ing)|expir(?:e|es|ed|ing|y|ation)|breach(?:ed|es)?|violat(?:e|es|ed|ion)|breakdown', re.I),
    'ending': re.compile(r'end(?:s|ed|ing)?|over|halt(?:s|ed|ing)?|stop(?:s|ped|ping)?|finish(?:es|ed|ing)?', re.I),
}


def canonical_url(value):
    parsed = urlsplit(str(value).strip())
    if parsed.scheme.lower() not in {'http', 'https'} or not parsed.netloc:
        raise ValueError('Every observed document needs an HTTP(S) source URL.')
    # Remove known tracking parameters, not identity-bearing ?id=... values.
    tracking = {'cmp', 'ref', 'fbclid', 'gclid', 'dclid', 'srsltid', 'guccounter'}
    query = [(key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=True)
             if not key.casefold().startswith('utm_') and key.casefold() not in tracking]
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip('/'), urlencode(sorted(query)), ''))


def matches(pattern, text):
    return sorted({m.group(0).casefold() for m in re.finditer(pattern, text, flags=re.I)})


def phrase_relations(text):
    """Expose nearest-target grammatical cues, never assert realized events.

    Actions attach to the nearest named target within six tokens in the same
    simple clause; ties favor a following target. Candidate targets and source
    spans remain visible. This is a transparent scope heuristic, not a parser.
    """
    relations = []
    for clause in CLAUSE_SPLIT.split(text):
        words = list(TOKEN.finditer(clause))
        tokens = [re.sub(r'[ -]', '', m.group(0).casefold()) for m in words]
        objects = [(i, TARGETS[token]) for i, token in enumerate(tokens) if token in TARGETS]
        for i, token in enumerate(tokens):
            kind = next((name for name, pattern in ACTION_PATTERNS.items() if pattern.fullmatch(token)), None)
            if kind is None:
                continue
            # "over Iran war" is generally a preposition, not an ending claim.
            if token == 'over' and (i == 0 or tokens[i - 1] not in
                                   {'is', 'are', 'be', 'been', 'almost', 'nearly', 'not', 'yet', 'war', 'conflict', 'hostilities'}):
                continue
            nearby = [(j, target) for j, target in objects if 0 < abs(i - j) <= 6]
            if not nearby:
                continue
            j, target = min(nearby, key=lambda pair: (abs(i - pair[0]), pair[0] < i, pair[0]))
            left, right = min(i, j), max(i, j)
            relations.append({'action_kind': kind, 'action_word': words[i].group(0),
                              'target': target, 'target_word': words[j].group(0),
                              'span': clause[words[left].start():words[right].end()],
                              'clause': clause.strip(),
                              'nearby_target_types': sorted({target_name for _, target_name in nearby}),
                              'method': 'nearest named target within six tokens in same simple clause'})
    return relations


def analyze_headline(headline):
    text = unicodedata.normalize('NFKC', str(headline or ''))
    # URL slugs can refer to an earlier headline and never determine these flags.
    text = re.sub(r'https?://\S+', '', text)
    legacy_entity = matches(LEGACY_ENTITY, text)
    entity = matches(EXPANDED_ENTITY, text)
    groups = {name: found for name, pattern in GROUPS.items() if (found := matches(pattern, text))}
    legacy = bool(legacy_entity and groups.get('legacy_conflict'))
    expanded = legacy or bool(entity and groups) or bool(groups.get('operation_code'))
    relations = phrase_relations(text)

    def relation(kind, target):
        return any(row['action_kind'] == kind and row['target'] == target for row in relations)

    negation = matches(NEGATION, text)
    conditional = matches(CONDITIONAL, text)
    quoted = bool(re.search(QUOTATION, text))
    attributed = matches(ATTRIBUTION, text)
    war_end = relation('ending', 'war')
    ceasefire_collapse = relation('collapse_expiry', 'ceasefire') or relation('ending', 'ceasefire')
    negated_end = any(row['action_kind'] == 'ending' and row['target'] == 'war'
                      and re.search(NEGATION, row['clause'], flags=re.I) for row in relations)
    negated_collapse = any(row['action_kind'] in {'collapse_expiry', 'ending'} and row['target'] == 'ceasefire'
                           and re.search(NEGATION, row['clause'], flags=re.I) for row in relations)
    labor = bool(re.search(LABOR, text, flags=re.I))
    trade_war = bool(re.search(TRADE_WAR, text, flags=re.I))
    scope_ambiguous = any(len(row['nearby_target_types']) > 1 for row in relations)
    reasons = []
    for present, reason in [(bool(negation), 'negation scope'), (bool(conditional), 'conditional/modal wording'),
                            (quoted or bool(attributed), 'quotation or attribution'),
                            (scope_ambiguous, 'multiple nearby action targets'),
                            (labor, 'possible labor meaning of strike'), (trade_war, 'possible nonphysical meaning of war'),
                            ('?' in text, 'question rather than assertion')]:
        if present:
            reasons.append(reason)
    return {
        'legacy_retrieved': legacy, 'expanded_retrieved': expanded, 'newly_retrieved': expanded and not legacy,
        'legacy_entity_matches': '|'.join(legacy_entity), 'expanded_entity_matches': '|'.join(entity),
        'matched_groups': '|'.join(groups), 'group_matches_json': json.dumps(groups, ensure_ascii=False, sort_keys=True),
        'operation_code_matches': '|'.join(groups.get('operation_code', [])),
        'ambiguous_term_matches': '|'.join(matches(AMBIGUOUS_TERMS, text)),
        'negation_terms': '|'.join(negation), 'conditional_modal_terms': '|'.join(conditional),
        'attribution_terms': '|'.join(attributed), 'negation_cue': bool(negation),
        'conditional_or_modal_cue': bool(conditional), 'quotation_cue': quoted, 'attribution_cue': bool(attributed),
        'code_name_cue': bool(groups.get('operation_code')), 'question_cue': '?' in text,
        'ceasefire_extension_cue': relation('extension', 'ceasefire'),
        'war_extension_cue': relation('extension', 'war'),
        'deadline_extension_cue': relation('extension', 'deadline'),
        'war_continuation_cue': relation('continuation', 'war'),
        'ceasefire_collapse_or_expiry_cue': ceasefire_collapse,
        'war_end_wording_cue': war_end,
        'negated_end_review': negated_end,
        'negated_collapse_review': negated_collapse,
        'labor_context_cue': labor, 'trade_war_cue': trade_war,
        'scope_ambiguity_cue': scope_ambiguous,
        'context_review_reasons': '; '.join(reasons),
        'phrase_relations_json': json.dumps(relations, ensure_ascii=False),
        'interpretation': 'Literal retrieval/scope cues only; no verified event, sentiment, probability or political evaluation',
    }


STRESS_CASES = [
    ('S01', 'Iran war continues', {'legacy_retrieved': True, 'war_continuation_cue': True}),
    ('S02', 'Tanker blocked in Strait of Hormuz', {'legacy_retrieved': False, 'expanded_retrieved': True}),
    ('S03', 'Operation Epic Fury enters a new phase', {'legacy_retrieved': False, 'expanded_retrieved': True, 'code_name_cue': True}),
    ('S04', 'Iran ceasefire extended for two weeks', {'ceasefire_extension_cue': True, 'war_extension_cue': False}),
    ('S05', 'Iran war extended for another month', {'war_extension_cue': True, 'ceasefire_extension_cue': False}),
    ('S06', 'Iran ceasefire collapses after missile attack', {'ceasefire_collapse_or_expiry_cue': True}),
    ('S07', 'Iran war is not over', {'war_end_wording_cue': True, 'negated_end_review': True}),
    ('S08', '"Iran war could end if deal holds", negotiator says', {'conditional_or_modal_cue': True, 'quotation_cue': True, 'attribution_cue': True, 'war_end_wording_cue': True}),
    ('S09', "Tehran's workers plan a transport strike", {'expanded_retrieved': True, 'labor_context_cue': True}),
    ('S10', 'Trade war with Iran escalates', {'legacy_retrieved': True, 'trade_war_cue': True}),
    ('S11', 'Ceasefire extension could prolong Iran war', {'ceasefire_extension_cue': True, 'war_extension_cue': True, 'conditional_or_modal_cue': True}),
    ('S12', 'Oil tanker hits cargo ship near Hormuz', {'expanded_retrieved': True, 'legacy_retrieved': False}),
    ('S13', 'UK nurses strike over pay', {'expanded_retrieved': False, 'labor_context_cue': True}),
    ('S14', 'Iran denies that the ceasefire has collapsed', {'negated_collapse_review': True}),
    ('S15', 'Iran war: ceasefire extended', {'ceasefire_extension_cue': True, 'war_extension_cue': False}),
    ('S16', 'Iran nuclear talks end without a deal as war fears grow', {'war_end_wording_cue': False, 'negation_cue': True}),
    ('S17', 'Iran deadline extended while war continues', {'deadline_extension_cue': True, 'war_extension_cue': False, 'war_continuation_cue': True}),
]


def stress_case_table():
    rows = []
    for case_id, headline, expected in STRESS_CASES:
        actual = analyze_headline(headline)
        selected = {key: actual[key] for key in expected}
        rows.append({'case_id': case_id, 'data_origin': 'synthetic software stress test',
                     'headline': headline, 'source_url': '',
                     'expected_rule_outputs_json': json.dumps(expected, sort_keys=True),
                     'actual_rule_outputs_json': json.dumps(selected, sort_keys=True),
                     'software_check_passed': expected == selected,
                     'warning': 'Author-constructed examples; passing checks are not empirical accuracy or human validation'})
    return pd.DataFrame(rows)


def audit_frame(frame):
    """Analyze current feature rows without imposing an additional fixed cutoff."""
    if not {'url', 'title'}.issubset(frame.columns):
        raise ValueError('news_features.csv must contain url and title.')
    source = frame.copy().fillna('')
    input_rows = len(source)
    future_excluded = 0
    if 'modified_after_cutoff' in source:
        future = source.modified_after_cutoff.astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])
        future_excluded = int(future.sum())
        source = source.loc[~future].copy()
    source['canonical_url'] = source.url.map(canonical_url)
    time_column = next((name for name in ['effective_utc', 'modified_utc', 'published_utc'] if name in source), None)
    if time_column:
        source['_sort_time'] = pd.to_datetime(source[time_column], errors='coerce', utc=True)
        source = source.sort_values('_sort_time', kind='stable', na_position='first')
    duplicate_rows = int(source.canonical_url.duplicated(keep='last').sum())
    source = source.drop_duplicates('canonical_url', keep='last')
    metadata = [name for name in ['url', 'canonical_url', 'title', 'archive_title', 'published_utc', 'modified_utc',
                                  'effective_utc', 'event_date', 'publication_event_date', 'war_relevant'] if name in source]
    rows = []
    for _, record in source.iterrows():
        headline = str(record.title).strip()
        basis = 'title'
        if not headline:
            headline = str(record.get('archive_title', '')).strip()
            basis = 'archive_title fallback' if headline else 'missing headline'
        rows.append({**{name: record[name] for name in metadata}, 'headline_analyzed': headline,
                     'headline_field': basis, 'data_origin': 'observed cached publisher headline',
                     **analyze_headline(headline)})
    documents = pd.DataFrame(rows)
    if documents.empty:
        raise ValueError('No eligible source documents remain for phrase audit.')
    term_documents = defaultdict(set)
    new_documents = set(documents.loc[documents.newly_retrieved, 'canonical_url'])
    for row in documents.itertuples(index=False):
        groups = json.loads(row.group_matches_json)
        groups['expanded_entity'] = row.expanded_entity_matches.split('|') if row.expanded_entity_matches else []
        groups['ambiguous_common_term'] = row.ambiguous_term_matches.split('|') if row.ambiguous_term_matches else []
        for group, terms in groups.items():
            for term in terms:
                term_documents[(group, term)].add(row.canonical_url)
    terms = pd.DataFrame([{'rule_group': group, 'matched_term': term, 'distinct_documents': len(urls),
                           'newly_retrieved_documents': len(urls & new_documents)}
                          for (group, term), urls in sorted(term_documents.items())],
                         columns=['rule_group', 'matched_term', 'distinct_documents', 'newly_retrieved_documents'])
    example_flags = ['newly_retrieved', 'code_name_cue', 'ceasefire_extension_cue', 'war_extension_cue',
                     'deadline_extension_cue', 'ceasefire_collapse_or_expiry_cue', 'negated_end_review',
                     'negated_collapse_review', 'conditional_or_modal_cue', 'quotation_cue',
                     'labor_context_cue', 'trade_war_cue', 'scope_ambiguity_cue']
    examples = []
    sort_columns = [name for name in ['published_utc', 'canonical_url'] if name in documents]
    ordered = documents.sort_values(sort_columns, kind='stable')
    for category in example_flags:
        for row in ordered.loc[ordered[category]].head(2).itertuples(index=False):
            words = row.headline_analyzed.split()
            snippet = ' '.join(words[:24]) + (' …' if len(words) > 24 else '')
            examples.append({'example_type': category, 'data_origin': 'observed cached publisher headline',
                             'headline_snippet': snippet, 'source_url': row.url,
                             'published_utc': getattr(row, 'published_utc', ''),
                             'legacy_retrieved': row.legacy_retrieved, 'expanded_retrieved': row.expanded_retrieved,
                             'context_review_reasons': row.context_review_reasons,
                             'phrase_relations_json': row.phrase_relations_json})
    summary = {
        'input_feature_rows': input_rows, 'modified_after_cutoff_rows_excluded': future_excluded,
        'duplicate_canonical_url_rows_removed': duplicate_rows, 'distinct_source_documents': len(documents),
        'missing_headlines': int(documents.headline_field.eq('missing headline').sum()),
        'legacy_retrieved_documents': int(documents.legacy_retrieved.sum()),
        'expanded_retrieved_documents': int(documents.expanded_retrieved.sum()),
        'newly_retrieved_distinct_documents': int(documents.newly_retrieved.sum()),
        'legacy_documents_lost': int((documents.legacy_retrieved & ~documents.expanded_retrieved).sum()),
        'cue_document_counts': {name: int(documents[name].sum()) for name in example_flags if name != 'newly_retrieved'},
        'latest_source_publication_utc': str(documents.published_utc.max()) if 'published_utc' in documents else None,
        'latest_source_effective_utc': str(documents.effective_utc.max()) if 'effective_utc' in documents else None,
        'legacy_rule': {'entity': LEGACY_ENTITY, 'context': LEGACY_CONFLICT,
                        'definition': 'Both must occur in the current headline. This is an audit comparator, not the supplied paper or existing body classifier.'},
        'expanded_rule': {'entity': EXPANDED_ENTITY, 'groups': GROUPS,
                          'definition': 'Retain every legacy match; otherwise require expanded entity plus a listed group, or an exact listed operation code.'},
        'novelty_definition': 'Newly retrieved relative to this comparator in this same saved corpus; not proof of newly coined wording or out-of-sample improvement.',
        'scope_method': 'Nearest named target within six tokens in a simple clause; all spans and candidate target types remain auditable. Negation/modals/quotes flag review, not truth values.',
        'example_selection': 'First two available documents chronologically per displayed cue; categories may repeat a document. Snippets are at most 24 words.',
        'accuracy_estimated': False, 'sentiment_scores_or_probabilities_produced': False,
        'limitations': ['Headline-only comparison is not interchangeable with the main headline-plus-body relevance rule.',
                       'Additional retrieval can include false positives, background nuclear/energy stories, questions, allegations, and nonphysical meanings of war/strike.',
                       'Exact operation names are code-name cues, not proof of a new/current military action; short fragments such as midnight or freedom are not code names.',
                       'Extending a ceasefire, extending a deadline and extending a war are different grammatical targets; no cue estimates war duration or ending probability.',
                       'A ceasefire described as collapsing, expiring or being violated is distinct from the war ending; negated and conditional wording still requires reading.',
                       'Source URLs are provenance only. Their slugs can be stale and are not retrieval text.',
                       'Canonical URLs count documents, not independent underlying events; paraphrases and repeated coverage remain.',
                       'No independently labeled evaluation corpus, inter-annotator agreement or empirical classifier accuracy is provided.'],
    }
    return documents, terms, pd.DataFrame(examples), summary


def build(root=ROOT):
    root = Path(root)
    path = root / 'data/processed/news_features.csv'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path, low_memory=False)
    documents, terms, examples, summary = audit_frame(frame)
    stress = stress_case_table()
    if not stress.software_check_passed.all():
        raise AssertionError('A synthetic software stress check failed; do not report it as empirical accuracy.')
    summary.update({'input_path': path.relative_to(root).as_posix(), 'input_sha256': digest,
                    'synthetic_software_checks': len(stress), 'synthetic_checks_all_passed': True,
                    'synthetic_checks_are_empirical_accuracy': False})
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Source feature file changed during the audit. Rerun after acquisition completes.')
    out = root / 'outputs'
    out.mkdir(parents=True, exist_ok=True)
    documents.to_csv(out / 'phrase_audit_documents.csv', index=False)
    terms.to_csv(out / 'phrase_audit_terms.csv', index=False)
    examples.to_csv(out / 'phrase_audit_examples.csv', index=False)
    stress.to_csv(out / 'phrase_audit_stress_cases.csv', index=False)
    (out / 'phrase_audit_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ['distinct_source_documents', 'legacy_retrieved_documents',
                                                 'expanded_retrieved_documents', 'newly_retrieved_distinct_documents',
                                                 'cue_document_counts', 'synthetic_software_checks']}, indent=2))
    return documents, terms, examples, summary


if __name__ == '__main__':
    build()
