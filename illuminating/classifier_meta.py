'''
A named collections of classfiers (modles) to apply
'''

CLASSIFIER_METAS = {

    "Topic" : [
        {
            "name": "TOPIC",
            "target_key": "illuminating_topic",
            "use_custom_vocab": False,
            "hit": None,
            "miss": None,
            "lexicon": "Topic_Lexicon.json",
        },
    ],

    "2020 Twitter CTA" : [
        {
            "name": "TW20CTA",
            "target_key": "illuminating_message_type",
            "use_custom_vocab": False,
            "hit": "call_to_action",
            "miss": None,
        },
    ],

    "2020 Facebook CTA" : [
        {
            "name": "FB20CTA",
            "target_key": "illuminating_message_type",
            "use_custom_vocab": False,
            "hit": "call_to_action",
            "miss": None,
        },
    ],

    "2020 Facebook Ad CTA Subtypes" : [
        {
            "name": "FB20_CTA_Voting",
            "target_key": "illuminating_cta_subtype",
            "use_custom_vocab": False,
            "hit": "voting",
            "miss": None,
        },
        {
            "name": "FB20_CTA_Fundraising",
            "target_key": "illuminating_cta_subtype",
            "use_custom_vocab": False,
            "hit": "fundraising",
            "miss": None,
        },
        {
            "name": "FB20_CTA_Engagement",
            "target_key": "illuminating_cta_subtype",
            "use_custom_vocab": False,
            "hit": "engagement",
            "miss": None,
        }
    ]
}
