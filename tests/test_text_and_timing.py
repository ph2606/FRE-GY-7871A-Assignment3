"""Meaningful regression checks for news measurement and the information cutoff.

Scoring fixtures contain physical shipping descriptions only. Attribution and
subject fixtures verify exclusion without evaluating an official or policy.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import news, text_features as tf


def test_assignment_one_weighting_and_empty_document():
    counts = [Counter({"DAMAGED": 3, "OIL": 3}), Counter({"OIL": 2}), Counter()]
    result = tf.score_counts(counts, {"negative": {"DAMAGED"}, "uncertainty": {"RISK"}})
    # Document frequency is one of three documents; average term frequency=3.
    assert result.loc[0, "negative_tfidf"] == pytest.approx(np.log(3))
    assert result.loc[0, "negative_pct"] == pytest.approx(50)
    assert result.loc[1, "negative_pct"] == 0
    assert pd.isna(result.loc[2, "negative_pct"])
    assert pd.isna(result.loc[2, "negative_tfidf"])


def test_inactive_lm_entries_are_not_category_members(tmp_path, monkeypatch):
    path = tmp_path / "data/raw/lexicons/LoughranMcDonald_MasterDictionary.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"Word": ["ACTIVE", "REMOVED", "NEUTRAL"], "Negative": [2009, -2018, 0],
                  "Positive": [0, 0, 0], "Uncertainty": [0, 0, 0]}).to_csv(path, index=False)
    monkeypatch.setattr(tf, "ROOT", tmp_path)
    monkeypatch.setattr(tf, "LM_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
    assert tf.lexicons()["negative"] == {"ACTIVE"}


@pytest.mark.parametrize("left,right", [('"', '"'), ('\u201c', '\u201d')])
def test_entire_multisentence_quotation_is_excluded(left, right):
    body = (f"The witness said: {left}Oil prices rose. Shipping was suspended. "
            f"The tanker sank.{right}\nOil deliveries resumed.")
    features, _ = tf.sentence_features("Iran oil shipping", body)
    assert "Shipping was suspended" not in features["physical_text"]
    assert "tanker sank" not in features["physical_text"]
    assert "Oil deliveries resumed" in features["physical_text"]


def test_month_may_is_not_a_threat_mention():
    features, _ = tf.sentence_features("Iran oil shipping", "Oil deliveries rose in May 2026.")
    assert features["threat_mentions"] == 0


def test_multparagraph_quote_does_not_reappear_after_attribution_exclusion():
    body = ('The minister said: "Oil shipments stopped.\n'
            'Shipping was suspended. The tanker sank."\n'
            'Oil deliveries resumed.')
    features, _ = tf.sentence_features("Iran oil shipping", body)
    assert features["physical_text"] == "Oil deliveries resumed."


def test_explicit_political_subject_is_excluded_without_scoring_it():
    features, _ = tf.sentence_features("Iran oil shipping", "The minister discussed oil.\nShipping resumed.")
    assert "minister" not in features["physical_text"]
    assert features["physical_text"] == "Shipping resumed."


@pytest.mark.parametrize("excluded", [
    "The White House believes the Iranian economy is as a result once again on the brink of collapse due to hyperinflation and lack of foreign exchange income caused by the US blockade of oil exports.",
    "The chief Iranian negotiator, Mohammad Baqer Ghalibaf, said the attack had killed families with children, and called the US the enemy.",
    'Two of the Iranian oil carriers were "disabled", the Pentagon said, while a third tanker, which was unladen, was destroyed.',
    "He warned that oil shipping would stop.",
    "Shipping was suspended, according to local witnesses.",
    "Sanctions restricted Iranian oil exports.",
])
def test_indirect_attribution_and_policy_are_excluded_before_counting(excluded):
    features, counts = tf.sentence_features("Iran oil shipping", excluded)
    assert features["war_relevant"]  # Attention remains independent of eligibility.
    assert features["n_physical_sentences"] == 0
    assert features["physical_text"] == ""
    assert not counts


def test_excluded_text_is_missing_rather_than_zero_sentiment():
    features, counts = tf.sentence_features("Iran oil shipping", "Our shipping deal was announced.")
    assert features["war_relevant"]
    result = tf.score_counts([counts], {"negative": {"DAMAGED"}})
    assert pd.isna(result.loc[0, "negative_pct"])
    assert pd.isna(result.loc[0, "negative_tfidf"])


def test_unattributed_physical_reporting_remains_eligible():
    features, counts = tf.sentence_features("Iran oil shipping", "Oil cargo spilled from a damaged tanker. Shipping resumed.")
    assert features["n_physical_sentences"] == 2
    assert counts["DAMAGED"] == 1


def test_paragraph_context_excludes_implicit_policy_assessment():
    body = ("The shipping move was premature. Sanctions were part of the strategy.\n"
            "Oil cargo spilled from a damaged tanker.")
    features, counts = tf.sentence_features("Iran oil shipping", body)
    assert features["physical_text"] == "Oil cargo spilled from a damaged tanker."
    assert "PREMATURE" not in counts


def test_session_mapping_accounts_for_dst_weekends_and_final_cutoff():
    sessions = pd.to_datetime(["2026-03-06", "2026-03-09", "2026-03-10"])
    stamps = pd.Series(pd.to_datetime([
        "2026-03-06T19:59:00Z",  # 14:59 EST -> Friday
        "2026-03-06T20:00:01Z",  # after 15 EST -> Monday
        "2026-03-08T12:00:00Z",  # Sunday after DST transition -> Monday
        "2026-03-09T18:59:00Z",  # 14:59 EDT -> Monday
        "2026-03-09T19:00:01Z",  # after 15 EDT -> Tuesday
        "2026-03-10T19:00:01Z",  # no observed later session
    ], utc=True))
    actual = tf.map_times_to_sessions(stamps, sessions)
    expected = pd.Series(pd.to_datetime(["2026-03-06", "2026-03-09", "2026-03-09",
                                         "2026-03-09", "2026-03-10", None]))
    pd.testing.assert_series_equal(actual, expected)


def test_exact_cutoff_is_inclusive_as_documented():
    sessions = pd.to_datetime(["2026-03-06", "2026-03-09"])
    stamps = pd.Series(pd.to_datetime(["2026-03-06T20:00:00Z"], utc=True))
    assert tf.map_times_to_sessions(stamps, sessions).iloc[0] == sessions[0]


def test_session_mapping_preserves_input_index():
    stamps = pd.Series(pd.to_datetime(["2026-06-12T21:00:00Z"], utc=True), index=["story"])
    result = tf.map_times_to_sessions(stamps, pd.to_datetime(["2026-06-12", "2026-06-15"]))
    assert result.index.tolist() == ["story"]
    assert result.loc["story"] == pd.Timestamp("2026-06-15")


def test_excluded_article_does_not_download(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Excluded content must not be downloaded")
    monkeypatch.setattr(news, "fetch", forbidden)
    record = {"url": "https://example.org/live/story", "exclusion": "live blog or multimedia"}
    assert news.article(record.copy()) == record


@pytest.mark.parametrize("stamp,expected", [
    ("2026-01-05T10:00:00", "publication timezone missing"),
    ("2025-12-31T23:59:59Z", "timestamp outside information set"),
    ("2026-09-17T00:00:00Z", "timestamp outside information set"),
    ("", "missing publication timestamp"),
])
def test_article_rejects_unusable_publication_clock(tmp_path, monkeypatch, stamp, expected):
    metadata = {"@type": "NewsArticle", "headline": "Iran oil shipping", "datePublished": stamp}
    body = "Oil shipping " + "cargo " * 85
    payload = (f'<script type="application/ld+json">{json.dumps(metadata)}</script>'
               f'<div data-gu-name="body"><p>{body}</p></div>').encode()
    monkeypatch.setattr(news, "ROOT", tmp_path)
    monkeypatch.setattr(news, "RAW", tmp_path / "data/raw/news")
    monkeypatch.setattr(news, "fetch", lambda *args: payload)
    monkeypatch.setattr(news.time, "sleep", lambda *args: None)
    result = news.article({"url": "https://example.org/story", "exclusion": ""})
    assert result["exclusion"] == expected


@pytest.fixture
def built_sample(tmp_path, monkeypatch):
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    (tmp_path / "outputs").mkdir()
    def document(name, words, hits, published, modified=None, excluded=""):
        body = "Oil " + "damaged " * hits + "cargo " * (words - hits - 1) + "."
        return {"url": "https://example.org/" + name, "title": "Iran oil shipping",
                "archive_title": "Iran oil shipping", "body": body, "exclusion": excluded,
                "published_utc": published, "modified_utc": modified or published}
    docs = [document("short", 20, 1, "2026-01-02T10:00:00Z"),
            document("long", 80, 1, "2026-01-02T11:00:00Z"),
            document("revised", 40, 2, "2026-01-02T12:00:00Z", "2026-01-06T10:00:00Z"),
            document("future", 30, 3, "2026-01-02T13:00:00Z", "2026-09-17T01:00:00Z"),
            document("excluded", 20, 4, "2026-01-05T12:00:00Z", excluded="exact duplicate body")]
    pd.DataFrame(docs).to_csv(processed / "news_documents.csv", index=False)
    sessions = pd.to_datetime(["2025-12-31", "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-09-16"])
    pd.DataFrame({"sp500": np.ones(len(sessions))}, index=sessions).to_csv(processed / "market_changes.csv")
    monkeypatch.setattr(tf, "ROOT", tmp_path)
    monkeypatch.setattr(tf, "lexicons", lambda: {"negative": {"DAMAGED"}, "positive": {"SAFE"}, "uncertainty": {"RISK"}})
    return tf.build()


def test_daily_percentages_are_token_weighted(built_sample):
    _, daily = built_sample
    # Two documents have 20 and 80 words, each with one lexical hit.
    assert daily.at[pd.Timestamp("2026-01-02"), "negative_pct"] == pytest.approx(2)


def test_no_news_is_missing_text_measure_not_neutrality(built_sample):
    _, daily = built_sample
    date = pd.Timestamp("2026-01-05")
    assert daily.at[date, "articles"] == 0
    assert daily.at[date, "war_articles"] == 0
    assert pd.isna(daily.at[date, "negative_pct"])
    assert pd.isna(daily.at[date, "d_negative_pct"])
    assert pd.isna(daily.at[pd.Timestamp("2026-01-06"), "d_negative_pct"])


def test_effective_date_uses_later_revision_clock(built_sample):
    features, _ = built_sample
    revised = features.set_index("url").loc["https://example.org/revised"]
    assert revised.event_date == pd.Timestamp("2026-01-06")
    assert revised.publication_event_date == pd.Timestamp("2026-01-02")


def test_future_revision_excluded_from_both_clocks(built_sample):
    _, daily = built_sample
    # The future body must not re-enter via an earlier publication timestamp.
    assert daily.at[pd.Timestamp("2026-01-02"), "articles_publication"] == 3


def test_corpus_has_no_artificial_precollection_zero_days(built_sample):
    _, daily = built_sample
    assert daily.index.min() >= pd.Timestamp("2026-01-01")
    assert pd.isna(daily.d_log_attention.iloc[0])
