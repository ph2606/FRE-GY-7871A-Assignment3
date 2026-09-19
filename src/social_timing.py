"""Publication timing and literal claim-type cues for retrieved Truth Social text.

Labels describe words in an attributed record, not verified military events,
sentiment, official/policy evaluations, or probabilities that a war will end.
Only ignored output files are written; existing source files are never changed.
"""
from pathlib import Path
import hashlib
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZONE = 'America/New_York'
CALENDAR_URL = 'https://www.nyse.com/trade/hours-calendars'
CALENDAR_CHECKED = '2026-09-19'
HOLIDAYS_2026 = {
    '2026-01-01': "New Year's Day", '2026-01-19': 'Martin Luther King Jr. Day',
    '2026-02-16': "Washington's Birthday", '2026-04-03': 'Good Friday',
    '2026-05-25': 'Memorial Day', '2026-06-19': 'Juneteenth',
    '2026-07-03': 'Independence Day observed', '2026-09-07': 'Labor Day',
    '2026-11-26': 'Thanksgiving', '2026-12-25': 'Christmas',
}
EARLY_CLOSES_2026 = {'2026-11-27': '13:00', '2026-12-24': '13:00'}
URL = re.compile(r'https?://[^\s<>]+', re.I)
# These overlap deliberately. In particular, cessation and continuation words
# can coexist; neither label means an event happened or predicts its outcome.
CLAIM_TYPES = {
    'ceasefire_truce': ('Ceasefire/truce wording', r'\b(?:cease[ -]?fire|truce)\b'),
    'negotiations_agreement': ('Negotiation/agreement wording', r'\b(?:negotiat\w*|talks?|discussions?|agreements?|deals?|diplomac\w*)\b'),
    'cessation_cancellation': ('Cessation/cancellation wording', r'\b(?:cancel(?:led|ed|ling|ing|lation)?|suspend(?:ed|ing|sion)?|paus(?:e|ed|ing)|hold\s+off|call(?:ed|ing)?\s+off|(?:end|ending|ended)\s+(?:of\s+)?(?:the\s+)?(?:war|conflict|hostilities)|(?:war|conflict|hostilities)\s+(?:is|are|will\s+be)\s+(?:over|ended))\b'),
    'continuation_extension': ('Continuation/extension wording', r'\b(?:extend(?:ed|ing|s)?|extensions?|continu(?:e|es|ed|ing|ation)|remain(?:s|ing|ed)?\s+(?:in\s+)?(?:full\s+force|effect)|blockade\s+(?:will\s+)?remain(?:s)?)\b'),
    'strikes_military': ('Strike/military-action wording', r'\b(?:attack\w*|strikes?|struck|bomb\w*|missiles?|drones?|military|naval|navy|epic\s+fury|midnight\s+hammer)\b'),
    'shipping_blockade': ('Shipping/blockade wording', r'\b(?:hormuz|straits?|shipping|ships?|tankers?|waterways?|navigation|blockad\w*|ports?)\b'),
    'nuclear': ('Nuclear wording', r'\b(?:nuclear|denuclear\w*|uranium|enrich\w*)\b'),
    'energy_prices': ('Energy/price wording', r'\b(?:oil|gas|gasoline|diesel|energy|crude|wti|brent|refiner\w*|pipeline\w*)\b'),
    'trade_sanctions': ('Trade/sanctions wording', r'\b(?:tariff\w*|sanction\w*|trade|trading|imports?|exports?)\b'),
    'humanitarian_civilian': ('Humanitarian/civilian wording', r'\b(?:humanitarian|civilians?|protest\w*|refugees?|hostages?|prisoners?|asylum)\b'),
}
CONDITION = re.compile(r'\b(?:if|unless|until|assuming|provided\s+that|subject\s+to|as\s+long\s+as)\b', re.I)
FUTURE = re.compile(r'\b(?:will|would|could|may|might|shall|going\s+to|scheduled)\b', re.I)


def nyse_schedule_2026():
    """Official scheduled equity hours, plus a prior-close boundary on Dec 31.

    December 31, 2025 is needed only for pre-open January 2 posts. There are no
    early closes inside the Jan 1--Sep 16 study period. July 2 is a full day.
    This schedule does not attempt to reconstruct unscheduled intraday halts.
    """
    dates = pd.bdate_range('2025-12-31', '2026-12-31')
    dates = dates[~dates.isin(pd.to_datetime(list(HOLIDAYS_2026)))]
    rows = []
    for date in dates:
        day = date.strftime('%Y-%m-%d')
        close = EARLY_CLOSES_2026.get(day, '16:00')
        rows.append({'session_date': date,
                     'open_et': pd.Timestamp(day + ' 09:30', tz=ZONE),
                     'close_et': pd.Timestamp(day + ' ' + close, tz=ZONE),
                     'scheduled_early_close': day in EARLY_CLOSES_2026})
    return pd.DataFrame(rows).set_index('session_date')


def claim_type_cues(text):
    """Return auditable literal cues in visible prose, excluding URL slugs.

    Quoted and reshared prose remains attributed to the archived record; this
    function does not infer that the account holder authored or endorses it.
    """
    body = URL.sub('', text or '').strip()
    matches = {}
    for key, (_, pattern) in CLAIM_TYPES.items():
        found = list(re.finditer(pattern, body, flags=re.I))
        if found:
            matches[key] = sorted({match.group(0) for match in found}, key=str.casefold)
    return {
        'claim_type_cues': '; '.join(CLAIM_TYPES[key][0] for key in matches) or 'No listed claim-type cue in non-URL text',
        'claim_type_matches_json': json.dumps(matches, ensure_ascii=False, sort_keys=True),
        'condition_cue_terms': '|'.join(sorted({m.group(0).lower() for m in CONDITION.finditer(body)})),
        'future_cue_terms': '|'.join(sorted({m.group(0).lower() for m in FUTURE.finditer(body)})),
        'claim_cue_text_basis': 'Visible text after URL removal; quotation/repost attribution is unresolved unless separately noted',
        **{'cue_' + key: key in matches for key in CLAIM_TYPES},
    }


def timing_fields(stamp, schedule, observed_sessions):
    """Map one timezone-aware timestamp without imputing a price observation.

    Core session is [open, close). Both reaction-close and tradable-open clocks
    use the first boundary strictly AFTER publication. A post at exactly a
    closing/opening auction is not assumed to have been observable before it.
    """
    stamp = pd.Timestamp(stamp)
    if stamp.tzinfo is None:
        raise ValueError('Publication timestamp must be timezone-aware.')
    stamp = stamp.tz_convert(ZONE)
    if stamp.year != 2026:
        raise ValueError('The verified calendar supports 2026 publication dates only.')
    day = stamp.tz_localize(None).normalize()
    observed = set(pd.DatetimeIndex(observed_sessions).normalize())
    sessions = pd.DatetimeIndex(schedule.index)
    opens = pd.DatetimeIndex(schedule.open_et)
    closes = pd.DatetimeIndex(schedule.close_et)
    is_session = day in sessions
    weekend = day.weekday() >= 5
    if is_session:
        row = schedule.loc[day]
        state = 'pre_open' if stamp < row.open_et else ('open' if stamp < row.close_et else 'after_close')
    else:
        row = None
        state = 'weekend' if weekend else 'holiday'

    close_i = int(closes.searchsorted(stamp, side='right'))
    open_i = int(opens.searchsorted(stamp, side='right'))
    after_date_i = int(sessions.searchsorted(day, side='right'))

    def date_at(i):
        return sessions[i].date().isoformat() if 0 <= i < len(sessions) else ''

    def stamp_at(values, i):
        return values[i].isoformat() if 0 <= i < len(values) else ''

    def observed_at(i):
        return bool(0 <= i < len(sessions) and sessions[i] in observed)

    return {
        'publication_et': stamp.isoformat(), 'publication_date_et': day.date().isoformat(),
        'publication_time_et': stamp.strftime('%H:%M:%S.%f').rstrip('0').rstrip('.'),
        'weekday_et': day.day_name(), 'utc_offset_hours': stamp.utcoffset().total_seconds() / 3600,
        'is_weekend': weekend, 'is_friday_to_sunday': day.weekday() in (4, 5, 6),
        'is_friday_after_close': day.weekday() == 4 and state == 'after_close',
        'is_weekend_or_friday_after_close': weekend or (day.weekday() == 4 and state == 'after_close'),
        'is_scheduled_session_date': is_session, 'is_observed_equity_session_date': day in observed,
        'cash_equity_state': state, 'cash_equity_open': state == 'open',
        'holiday_name': HOLIDAYS_2026.get(day.date().isoformat(), ''),
        'scheduled_open_et': row.open_et.isoformat() if row is not None else '',
        'scheduled_close_et': row.close_et.isoformat() if row is not None else '',
        'previous_close_et': stamp_at(closes, close_i - 1),
        'reaction_close_session': date_at(close_i), 'first_close_after_post_et': stamp_at(closes, close_i),
        'reaction_close_has_observed_price': observed_at(close_i),
        'next_open_session': date_at(open_i), 'first_open_after_post_et': stamp_at(opens, open_i),
        'next_open_has_observed_price': observed_at(open_i),
        'next_us_session_after_post_date': date_at(after_date_i),
        'following_reaction_session': date_at(close_i + 1),
        'following_reaction_has_observed_price': observed_at(close_i + 1),
        'minutes_to_reaction_close': (closes[close_i] - stamp).total_seconds() / 60 if close_i < len(closes) else None,
        'minutes_to_next_open': (opens[open_i] - stamp).total_seconds() / 60 if open_i < len(opens) else None,
    }


def daily_counts(audit, schedule, observed_sessions):
    """Factual posting-presence indicators: zero means no retrieved record.

    Calendar rows and session rows are deliberately separate. A zero is not a
    statement that there was no war news, omitted media post, or conflict risk.
    """
    calendar = pd.DataFrame(index=pd.date_range('2026-01-01', '2026-09-16'))
    calendar.index.name = 'date_et'
    day = pd.to_datetime(audit.publication_date_et)
    calendar['candidate_posts'] = audit.groupby(day).size().reindex(calendar.index, fill_value=0)
    calendar['candidate_post_day'] = calendar.candidate_posts.gt(0).astype(int)
    calendar['scheduled_equity_session'] = calendar.index.isin(schedule.index)
    calendar['observed_equity_session'] = calendar.index.isin(observed_sessions)
    for state in ['pre_open', 'open', 'after_close', 'weekend', 'holiday']:
        calendar[state + '_posts'] = audit.cash_equity_state.eq(state).groupby(day).sum().reindex(calendar.index, fill_value=0)
    for key in CLAIM_TYPES:
        calendar['cue_' + key + '_posts'] = audit['cue_' + key].groupby(day).sum().reindex(calendar.index, fill_value=0)

    observed = pd.DatetimeIndex(observed_sessions)
    observed = observed[(observed >= '2026-01-01') & (observed <= '2026-09-16')]
    session = pd.DataFrame(index=observed)
    session.index.name = 'session_date'
    for column, prefix in [('reaction_close_session', 'close_window'), ('next_open_session', 'before_next_open')]:
        group = pd.to_datetime(audit[column].replace('', pd.NA))
        session[prefix + '_posts'] = audit.groupby(group).size().reindex(session.index, fill_value=0)
        session[prefix + '_post_presence'] = session[prefix + '_posts'].gt(0).astype(int)
        for key in CLAIM_TYPES:
            session[prefix + '_cue_' + key + '_posts'] = audit['cue_' + key].groupby(group).sum().reindex(session.index, fill_value=0)
    reaction = pd.to_datetime(audit.reaction_close_session.replace('', pd.NA))
    for column in ['is_weekend', 'is_friday_after_close', 'is_friday_to_sunday']:
        session['close_window_' + column + '_posts'] = audit[column].groupby(reaction).sum().reindex(session.index, fill_value=0)
    return calendar, session


def build(root=ROOT):
    root = Path(root)
    sources = {
        'candidates': root / 'data/processed/social_candidates.csv',
        'attribution_audit': root / 'outputs/social_relevance_audit.csv',
        'market_levels': root / 'data/processed/market_levels.csv',
    }
    source_hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources.values()}
    candidates = pd.read_csv(sources['candidates'], dtype=str).fillna('')
    attribution = pd.read_csv(sources['attribution_audit'], dtype=str).fillna('')
    if candidates.id.duplicated().any() or attribution.id.duplicated().any():
        raise ValueError('Source IDs must be unique and preserved as strings.')
    if set(candidates.id) != set(attribution.id):
        raise ValueError('Run src.social_audit: attribution and candidate ID sets differ.')
    check = candidates.set_index('id').text.eq(attribution.set_index('id').text.reindex(candidates.id))
    if not check.all():
        raise ValueError('Attribution audit text differs from candidate source text.')
    if set(candidates.platform) != {'Truth Social'}:
        raise ValueError('Do not combine platforms in this Truth Social audit.')
    levels = pd.read_csv(sources['market_levels'], index_col=0, parse_dates=True)
    observed = pd.DatetimeIndex(levels.index[levels['^GSPC_close'].notna()]).normalize().sort_values().unique()
    schedule = nyse_schedule_2026()
    observed_2026 = observed[observed.year == 2026]
    unexpected = observed_2026.difference(schedule.index)
    if len(unexpected):
        raise ValueError(f'Observed equity prices on scheduled non-session dates: {unexpected.tolist()}')
    expected = schedule.index[(schedule.index >= observed_2026.min()) & (schedule.index <= observed_2026.max())]
    missing = expected.difference(observed_2026)
    audit_columns = ['id', 'source_form', 'relevance_match_location', 'authorship_audit', 'media_count',
                     'media_text_not_reviewed', 'explicit_repost_marker', 'explicit_attribution_prefix', 'qualitative_context_note']
    result = candidates.merge(attribution[audit_columns], on='id', how='left', validate='one_to_one')
    additions = []
    for row in result.itertuples(index=False):
        timing = timing_fields(row.created_at_utc, schedule, observed)
        if pd.Timestamp(row.created_at_et) != pd.Timestamp(row.created_at_utc) or row.date_et != timing['publication_date_et']:
            raise ValueError(f'Original timestamp/date inconsistency: {row.id}')
        additions.append({**timing, **claim_type_cues(row.text)})
    audit = pd.concat([result.reset_index(drop=True), pd.DataFrame(additions)], axis=1)
    calendar, session = daily_counts(audit, schedule, observed)
    summary = {
        'candidate_records': len(audit), 'platform': 'Truth Social',
        'publication_window_et': '2026-01-01 through 2026-09-16 inclusive',
        'input_sha256': source_hashes,
        'calendar_source': CALENDAR_URL, 'calendar_verified_date': CALENDAR_CHECKED,
        'core_hours_et': '09:30 <= timestamp < 16:00; scheduled early closes use 13:00',
        'holidays_2026': HOLIDAYS_2026, 'early_closes_2026': EARLY_CLOSES_2026,
        'early_closes_in_observation_window': [],
        'observed_session_column': '^GSPC_close',
        'missing_expected_price_sessions': [x.date().isoformat() for x in missing],
        'unexpected_closed_date_observations': [],
        'equity_state_counts': {k: int(v) for k, v in audit.cash_equity_state.value_counts().items()},
        'weekday_counts': {k: int(v) for k, v in audit.weekday_et.value_counts().items()},
        'friday_to_sunday_posts': int(audit.is_friday_to_sunday.sum()),
        'weekend_posts': int(audit.is_weekend.sum()),
        'friday_after_close_posts': int(audit.is_friday_after_close.sum()),
        'reaction_close_without_observed_price': int((~audit.reaction_close_has_observed_price).sum()),
        'next_open_without_observed_price': int((~audit.next_open_has_observed_price).sum()),
        'calendar_days_with_candidate_post': int(calendar.candidate_post_day.sum()),
        'observed_close_windows_with_candidate_post': int(session.close_window_post_presence.sum()),
        'claim_cue_record_counts': {key: int(audit['cue_' + key].sum()) for key in CLAIM_TYPES},
        'claim_type_rules': {key: {'label': label, 'regex': pattern} for key, (label, pattern) in CLAIM_TYPES.items()},
        'reaction_clock': 'First scheduled close strictly after publication; Friday after-close/weekend/holiday material maps to the next scheduled close.',
        'open_clock': 'First scheduled open strictly after publication; intraday posts enter the following scheduled opening window.',
        'next_day_clock': 'following_reaction_session is the next scheduled session after reaction_close_session; price availability is separately flagged.',
        'zero_one_definition': 'Posting-presence indicators equal 1 if at least one retrieved candidate is in the specified window; 0 otherwise. They are not war-news absence, sentiment, or probability labels.',
        'attribution': 'Claim types are nonexclusive literal wording cues after URL removal; quotes/reposts are not assigned verified authorship. No claim is independently verified as an event by this module.',
        'limitations': [f'{len(audit)} records form a text retrieval set, not a complete Iran-post census or verified X archive.',
                       'Media, implicit references, linked page contents, deletions and archive gaps remain.',
                       'Scheduled cash-equity hours do not describe oil futures, Treasury, FX, foreign exchanges or unscheduled intraday halts.',
                       'Daily closing observations cannot estimate an overnight versus intraday return response; opening prices are not provided by this module.',
                       'Reposts and repeated claims can be old information; posting time is not first-public-news time.',
                       'Literal cessation/continuation words do not establish an end to hostilities, a forecast, or a sentiment judgment.'],
        'sentiment_scores_probabilities_or_rankings_produced': False,
    }
    out = root / 'outputs'
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / 'social_timing_audit.csv', index=False)
    calendar.to_csv(out / 'social_timing_daily.csv')
    session.to_csv(out / 'social_session_timing.csv')
    (out / 'social_timing_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    for name, digest in source_hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Input changed during audit: {name}')
    print(json.dumps({key: summary[key] for key in ['candidate_records', 'equity_state_counts', 'friday_to_sunday_posts',
                                                  'reaction_close_without_observed_price', 'claim_cue_record_counts']}, indent=2))
    return audit, calendar, session, summary


if __name__ == '__main__':
    build()
