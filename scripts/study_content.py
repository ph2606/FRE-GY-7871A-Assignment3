"""Shared research narrative and displays for the report and executed notebook."""
from pathlib import Path
import json
import pandas as pd

SOCIAL_EXAMPLES=[
 ('2026-01-02','115824439366264186','Conditional intervention tied to treatment of protesters. Future tense and the stated condition do not establish that intervention occurred.'),
 ('2026-01-12','115884319075881590','A tariff announcement concerns trade restrictions, a separate channel from a physical interruption of crude supply.'),
 ('2026-02-04','116013105630663812','Iran is one subject of a conversation with China’s president. Favorable language describes the conversation and relationship, rather than an observed end to conflict.'),
 ('2026-03-01','116152251973821428','A conditional response to an anticipated Iranian action. Preserve the condition, future tense and emphasis when interpreting the statement.'),
 ('2026-04-01','116329512466946656','A claimed ceasefire request, a Hormuz condition and continuing-attack language coexist. These propositions concern different outcomes and should be read separately.'),
 ('2026-05-03','116512555123589170','A shipping-operation announcement appears alongside negotiation references. Navigation access and stated conditions matter for the oil-supply channel.'),
 ('2026-08-01','117023461141824050','A claimed cancellation of a planned attack is linked to an agreement condition. It is not an unconditional statement that the conflict has ended.'),
 ('2026-09-01','117196950497702512','The account reports strikes around Hormuz and states a condition for a further response. Keep the reported claim distinct from the prospective condition.'),
 ('2026-09-07','117232304514057261','A forecast about oil and gasoline prices depends on a future war outcome. It is an attributed price claim, not an observed market effect.'),
]

READINGS=[
 ('Course presentation, undated', 'Twitter sentiment analysis overview', 'Keyword retrieval is a sampling rule; historical API instructions do not guarantee complete current archives.'),
 ('Yalin Yener, 2020', 'Step by Step: Twitter Sentiment Analysis in Python', 'Keep original text and a separate normalized field; distinguish reposts from duplicate IDs; vocabulary clouds do not validate sentiment.'),
 ('Bagheri and Islam, 2017', 'Sentiment analysis of twitter data', 'Short text, abbreviations and context challenge dictionary methods; query proportions are not accuracy estimates.'),
 ('Psomakelis et al., 2014', 'Comparing Methods for Twitter Sentiment Analysis', 'Representation and contextual features matter; their manually labeled benchmark does not establish accuracy on Iran-war statements.'),
 ('Carvalho and Plastino, 2021; online 2020', 'On the evaluation and combination of state-of-the-art features in Twitter sentiment analysis', 'Preserve negation and expressive features; evaluate feature combinations and domain transfer. Their binary benchmarks do not resolve neutral or mixed war text.'),
 ('Supplied presentation, undated', 'Exploring Differences in the Sentiment Analysis Tools using Twitter Data concerning Autism Awareness', 'Separate VADER from trained classifiers and use human validation. TN/(FP+TN) is specificity, not precision; VADER proportions are not probabilities.'),
 ('Stanford-derived course slides, undated', 'Sentiment Analysis', 'Identify the attitude holder, target and textual unit; irony and mixed targets make a generic positive/negative label unsuitable for war duration.'),
]


from study_sections import sections
