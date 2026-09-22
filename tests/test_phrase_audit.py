"""Synthetic linguistic stress checks and observed-row provenance invariants.

These are software tests with constructed inputs, not measured NLP accuracy.
"""
import json

import pandas as pd
import pytest

from src.phrase_audit import STRESS_CASES, analyze_headline, audit_frame, canonical_url, stress_case_table


@pytest.mark.parametrize('case_id,headline,expected', STRESS_CASES)
def test_synthetic_phrase_scope(case_id, headline, expected):
    actual = analyze_headline(headline)
    assert {key: actual[key] for key in expected} == expected, case_id


def test_short_codename_fragments_do_not_retrieve_unrelated_headlines():
    assert not analyze_headline('Doomsday Clock moves closer to midnight')['code_name_cue']
    assert not analyze_headline('Freedom of speech debated in parliament')['expanded_retrieved']


def test_url_slug_never_supplies_missing_headline_entities():
    result = analyze_headline('Markets rebound https://example.org/iran-war-ceasefire-extended')
    assert not result['legacy_retrieved']
    assert not result['expanded_retrieved']
    assert not result['ceasefire_extension_cue']


def test_trade_and_labor_examples_are_candidates_with_explicit_ambiguity():
    trade = analyze_headline('Trade war with Iran continues')
    labor = analyze_headline('Tehran workers announce strike')
    assert trade['expanded_retrieved'] and trade['trade_war_cue']
    assert labor['expanded_retrieved'] and labor['labor_context_cue']
    assert 'nonphysical' in trade['context_review_reasons']
    assert 'labor' in labor['context_review_reasons']


def test_distinct_documents_use_canonical_urls_and_latest_headline():
    frame = pd.DataFrame([
        {'url': 'https://example.org/a?utm_source=old', 'title': 'Iran war continues', 'effective_utc': '2026-09-16T10:00:00Z'},
        {'url': 'https://example.org/a#latest', 'title': 'Tanker blocked in Strait of Hormuz', 'effective_utc': '2026-09-18T11:00:00Z'},
        {'url': 'https://example.org/b', 'title': 'Iran ceasefire extended', 'effective_utc': '2026-09-18T10:00:00Z'},
    ])
    documents, _, _, summary = audit_frame(frame)
    assert summary['distinct_source_documents'] == 2
    assert summary['duplicate_canonical_url_rows_removed'] == 1
    assert summary['newly_retrieved_distinct_documents'] == 2
    assert documents.loc[documents.canonical_url.eq('https://example.org/a'), 'headline_analyzed'].iloc[0] == 'Tanker blocked in Strait of Hormuz'
    assert summary['latest_source_effective_utc'] == '2026-09-18T11:00:00Z'


def test_cutoff_is_inherited_from_input_not_frozen_to_old_calendar_date():
    frame = pd.DataFrame([
        {'url': 'https://example.org/a', 'title': 'Iran war continues', 'published_utc': '2026-09-18T10:00:00Z', 'modified_after_cutoff': False},
        {'url': 'https://example.org/b', 'title': 'Iran ceasefire extended', 'published_utc': '2026-09-19T10:00:00Z', 'modified_after_cutoff': True},
    ])
    documents, _, _, summary = audit_frame(frame)
    assert len(documents) == 1
    assert summary['modified_after_cutoff_rows_excluded'] == 1
    assert summary['latest_source_publication_utc'] == '2026-09-18T10:00:00Z'


def test_synthetic_examples_have_no_invented_source_or_accuracy_claim():
    stress = stress_case_table()
    assert stress.software_check_passed.all()
    assert stress.data_origin.eq('synthetic software stress test').all()
    assert stress.source_url.eq('').all()
    assert stress.warning.str.contains('not empirical accuracy').all()


def test_relation_evidence_preserves_exact_phrase_and_different_targets():
    result = analyze_headline('Ceasefire extension could prolong Iran war')
    evidence = json.loads(result['phrase_relations_json'])
    assert {row['target'] for row in evidence if row['action_kind'] == 'extension'} == {'ceasefire', 'war'}
    assert 'prolong Iran war' in {row['span'] for row in evidence}


def test_separate_comma_clauses_keep_ending_target_and_negation():
    result = analyze_headline('Iran war will not end, ceasefire is extended')
    assert result['war_end_wording_cue']
    assert result['negated_end_review']
    assert result['ceasefire_extension_cue']
    assert not result['negated_collapse_review']


def test_negation_in_another_clause_does_not_negate_war_ending():
    result = analyze_headline('Iran war ends; oil prices do not fall')
    assert result['war_end_wording_cue']
    assert result['negation_cue']
    assert not result['negated_end_review']


def test_unspaced_dash_separates_ending_and_ceasefire_extension():
    result = analyze_headline('Iran war ends—ceasefire extended')
    assert result['war_end_wording_cue']
    assert result['ceasefire_extension_cue']
    assert not result['ceasefire_collapse_or_expiry_cue']


def test_url_identity_parameters_are_retained_but_tracking_is_removed():
    assert canonical_url('https://example.org/article?id=1&utm_source=x#section') == 'https://example.org/article?id=1'
    assert canonical_url('https://example.org/article?id=2') != canonical_url('https://example.org/article?id=1')


def test_numeric_cutoff_flags_are_respected():
    frame = pd.DataFrame([
        {'url': 'https://example.org/a', 'title': 'Iran war continues', 'modified_after_cutoff': 0.0},
        {'url': 'https://example.org/b', 'title': 'Iran ceasefire extended', 'modified_after_cutoff': 1.0},
    ])
    documents, _, _, summary = audit_frame(frame)
    assert len(documents) == 1
    assert summary['modified_after_cutoff_rows_excluded'] == 1
