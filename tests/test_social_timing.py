"""Clock-boundary and attribution-sensitive checks, without live downloads."""
import pandas as pd
import pytest

from src.social_timing import claim_type_cues, daily_counts, nyse_schedule_2026, timing_fields


@pytest.fixture
def schedule():
    return nyse_schedule_2026()


def classify(timestamp, schedule, observed=None):
    if observed is None:
        observed = schedule.index[schedule.index <= '2026-09-16']
    return timing_fields(timestamp, schedule, observed)


def test_exact_open_and_close_do_not_assume_preauction_availability(schedule):
    opening = classify('2026-06-11T09:30:00-04:00', schedule)
    assert opening['cash_equity_state'] == 'open'
    assert opening['reaction_close_session'] == '2026-06-11'
    assert opening['next_open_session'] == '2026-06-12'
    closing = classify('2026-06-11T16:00:00-04:00', schedule)
    assert closing['cash_equity_state'] == 'after_close'
    assert closing['reaction_close_session'] == '2026-06-12'
    just_before = classify('2026-06-11T15:59:59.999999-04:00', schedule)
    assert just_before['reaction_close_session'] == '2026-06-11'


def test_friday_to_sunday_mapping_respects_monday_holiday(schedule):
    for timestamp in ['2026-09-04T16:01:00-04:00', '2026-09-05T11:00:00-04:00', '2026-09-06T23:00:00-04:00']:
        result = classify(timestamp, schedule)
        assert result['is_friday_to_sunday']
        assert result['is_weekend_or_friday_after_close']
        assert result['reaction_close_session'] == '2026-09-08'
        assert result['next_open_session'] == '2026-09-08'


def test_july_second_is_full_session_third_is_holiday(schedule):
    result = classify('2026-07-02T14:30:00-04:00', schedule)
    assert result['cash_equity_state'] == 'open'
    assert result['reaction_close_session'] == '2026-07-02'
    holiday = classify('2026-07-03T10:00:00-04:00', schedule)
    assert holiday['cash_equity_state'] == 'holiday'
    assert holiday['reaction_close_session'] == '2026-07-06'


def test_documented_early_close_and_dst_are_applied(schedule):
    result = classify('2026-11-27T18:00:00Z', schedule)
    assert result['publication_time_et'] == '13:00:00'
    assert result['cash_equity_state'] == 'after_close'
    assert result['reaction_close_session'] == '2026-11-30'
    assert result['scheduled_close_et'] == '2026-11-27T13:00:00-05:00'
    assert classify('2026-03-06T14:00:00Z', schedule)['cash_equity_state'] == 'pre_open'
    assert classify('2026-03-09T14:00:00Z', schedule)['cash_equity_state'] == 'open'


def test_post_cutoff_session_is_scheduled_without_inventing_market_data(schedule):
    result = classify('2026-09-16T16:01:00-04:00', schedule)
    assert result['reaction_close_session'] == '2026-09-17'
    assert not result['reaction_close_has_observed_price']
    assert result['following_reaction_session'] == '2026-09-18'
    assert not result['following_reaction_has_observed_price']


def test_missing_price_does_not_change_scheduled_open_state(schedule):
    observed = schedule.index[schedule.index != pd.Timestamp('2026-06-11')]
    result = classify('2026-06-11T13:00:00-04:00', schedule, observed)
    assert result['cash_equity_state'] == 'open'
    assert not result['is_observed_equity_session_date']
    assert not result['reaction_close_has_observed_price']


def test_timezone_is_required_and_first_january_preopen_has_previous_close(schedule):
    with pytest.raises(ValueError, match='timezone-aware'):
        classify('2026-06-11 13:00', schedule)
    result = classify('2026-01-02T07:58:08.615Z', schedule)
    assert result['publication_date_et'] == '2026-01-02'
    assert result['previous_close_et'] == '2025-12-31T16:00:00-05:00'


def test_url_slug_does_not_create_claim_types():
    result = claim_type_cues('https://example.org/iran-ceasefire-cancelled-strikes')
    assert not any(value for key, value in result.items() if key.startswith('cue_'))


def test_quoted_conditional_and_concurrent_claims_are_not_forced_into_one_event():
    result = claim_type_cues('RT @other: "If a deal is signed, strikes will be cancelled. The blockade will remain in full force until then."')
    assert result['cue_negotiations_agreement']
    assert result['cue_cessation_cancellation']
    assert result['cue_continuation_extension']
    assert result['cue_strikes_military']
    assert result['cue_shipping_blockade']
    assert result['condition_cue_terms'] == 'if|until'
    assert 'unresolved' in result['claim_cue_text_basis']
    assert not any('probability' in key or 'sentiment_score' in key for key in result)


def test_calendar_day_and_session_presence_preserve_different_information_windows(schedule):
    stamps = ['2026-09-04T16:01:00-04:00', '2026-09-05T11:00:00-04:00',
              '2026-09-06T23:00:00-04:00', '2026-09-07T12:00:00-04:00',
              '2026-09-08T13:00:00-04:00']
    observed = schedule.index[schedule.index <= '2026-09-16']
    rows = [{**classify(stamp, schedule, observed), **claim_type_cues('Iran shipping continues.')} for stamp in stamps]
    calendar, sessions = daily_counts(pd.DataFrame(rows), schedule, observed)
    assert calendar.candidate_posts.sum() == 5
    assert calendar.candidate_post_day.sum() == 5
    assert sessions.loc['2026-09-08', 'close_window_posts'] == 5
    assert sessions.loc['2026-09-08', 'close_window_post_presence'] == 1
    assert sessions.loc['2026-09-08', 'before_next_open_posts'] == 4
    assert sessions.loc['2026-09-09', 'before_next_open_posts'] == 1
    assert sessions.loc['2026-09-08', 'close_window_is_weekend_posts'] == 2
    assert sessions.loc['2026-09-08', 'close_window_is_friday_after_close_posts'] == 1
    assert sessions.before_next_open_posts.sum() == 5
