import re
import string
from collections import Counter

# ─────────────────────────────────────────────
# AGENT 1: EMOTION & MANIPULATION DETECTOR
# ─────────────────────────────────────────────

FEAR_WORDS = [
    "shocking", "terrifying", "horrifying", "disaster", "crisis", "danger",
    "threat", "warning", "alert", "panic", "emergency", "catastrophe",
    "deadly", "fatal", "killer", "destroy", "collapse", "chaos", "invasion"
]

ANGER_WORDS = [
    "outrage", "furious", "disgusting", "corrupt", "evil", "criminal",
    "traitor", "liar", "fraud", "scam", "stolen", "rigged", "cheat",
    "betray", "enemy", "radical", "extremist", "puppet", "globalist"
]

MANIPULATION_WORDS = [
    "they don't want you to know", "mainstream media won't tell",
    "the truth is", "wake up", "sheeple", "deep state", "new world order",
    "they are hiding", "secret agenda", "coverup", "suppressed",
    "what they don't tell you", "banned", "censored", "share before deleted"
]

SENSATIONAL_WORDS = [
    "breaking", "urgent", "exclusive", "bombshell", "explosive", "shocking",
    "unbelievable", "incredible", "mind-blowing", "jaw-dropping", "exposed",
    "scandal", "leaked", "whistleblower", "must read", "must watch",
    "share this", "going viral", "everyone is talking"
]

def emotion_agent(text):
    """
    Detects emotional manipulation in text.
    Returns a dict with scores and detected phrases.
    """
    text_lower = text.lower()
    words = text_lower.split()

    fear_found         = [w for w in FEAR_WORDS if w in text_lower]
    anger_found        = [w for w in ANGER_WORDS if w in text_lower]
    manipulation_found = [p for p in MANIPULATION_WORDS if p in text_lower]
    sensational_found  = [w for w in SENSATIONAL_WORDS if w in text_lower]

    # Count exclamation marks and all caps words
    exclamation_count = text.count('!')
    caps_words = [w for w in text.split() if w.isupper() and len(w) > 2]

    # Calculate emotion score (0-10)
    raw_score = (
        len(fear_found) * 1.5 +
        len(anger_found) * 1.5 +
        len(manipulation_found) * 3.0 +
        len(sensational_found) * 1.0 +
        exclamation_count * 0.5 +
        len(caps_words) * 0.3
    )
    emotion_score = min(10, round(raw_score, 1))

    if emotion_score >= 7:
        level = "HIGH"
    elif emotion_score >= 4:
        level = "MEDIUM"
    else:
        level = "LOW"

    detected = fear_found + anger_found + manipulation_found + sensational_found

    return {
        "emotion_score":    emotion_score,
        "emotion_level":    level,
        "fear_words":       fear_found,
        "anger_words":      anger_found,
        "manipulation":     manipulation_found,
        "sensational":      sensational_found,
        "exclamations":     exclamation_count,
        "caps_words":       caps_words[:5],
        "detected_phrases": list(set(detected))[:8],
    }


# ─────────────────────────────────────────────
# AGENT 2: CREDIBILITY SCORER
# ─────────────────────────────────────────────

CREDIBLE_SOURCES = [
    "reuters", "bbc", "associated press", "ap news", "the hindu",
    "times of india", "ndtv", "bloomberg", "washington post",
    "new york times", "guardian", "npr", "pbs", "abc news", "cbs news"
]

SUSPICIOUS_PATTERNS = [
    r'\b(100%|proven|guaranteed|confirmed|officials confirm)\b',
    r'\b(miracle|cure|secret|hidden|they hide)\b',
    r'\b(share this|spread the word|tell everyone)\b',
    r'\b(anonymous source|insider|whistleblower)\b',
    r'(!!!|\?\?\?)',
]

CLICK_BAIT_PATTERNS = [
    "you won't believe", "what happened next", "this is why",
    "the reason will shock", "number \d will surprise",
    "doctors hate", "one weird trick", "this simple trick"
]

def credibility_agent(text, title=""):
    """
    Scores the credibility of a news article.
    Returns a credibility score out of 10.
    """
    text_lower  = (title + " " + text).lower()
    full_text   = title + " " + text

    score = 10  # start at 10, deduct for red flags

    # Check for credible source mentions
    source_found = [s for s in CREDIBLE_SOURCES if s in text_lower]
    if source_found:
        score += 2

    # Check suspicious patterns
    suspicious_hits = []
    for pattern in SUSPICIOUS_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            suspicious_hits.extend(matches)
            score -= 1.5

    # Check clickbait
    clickbait_found = [p for p in CLICK_BAIT_PATTERNS if p in text_lower]
    score -= len(clickbait_found) * 1.5

    # Check excessive punctuation
    exclamations = full_text.count('!')
    questions    = full_text.count('?')
    if exclamations > 3:
        score -= 1
    if questions > 5:
        score -= 0.5

    # Check ALL CAPS words (shouting)
    caps_ratio = len([w for w in full_text.split() if w.isupper() and len(w) > 2])
    if caps_ratio > 5:
        score -= 1.5

    # Check article length (very short = suspicious)
    word_count = len(text.split())
    if word_count < 50:
        score -= 2
    elif word_count > 300:
        score += 1

    # Check for quotes (credible articles quote sources)
    quote_count = full_text.count('"')
    if quote_count >= 2:
        score += 0.5

    score = max(0, min(10, round(score, 1)))

    if score >= 7:
        level = "HIGH"
    elif score >= 4:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "credibility_score":   score,
        "credibility_level":   level,
        "credible_sources":    source_found,
        "suspicious_patterns": [str(s) for s in suspicious_hits[:5]],
        "clickbait_phrases":   clickbait_found,
        "word_count":          word_count,
        "has_quotes":          quote_count >= 2,
    }


# ─────────────────────────────────────────────
# AGENT 3: CLAIM EXTRACTOR
# ─────────────────────────────────────────────

CLAIM_INDICATORS = [
    "claims that", "according to", "states that", "reveals that",
    "confirms that", "proves that", "shows that", "says that",
    "alleges that", "reports that", "announces that", "admits that"
]

FACTUAL_CLAIM_PATTERNS = [
    r'\b\d+%\b',                          # percentages
    r'\b\d{4}\b',                         # years
    r'\$[\d,]+',                          # money amounts
    r'\b\d+ (million|billion|thousand)\b',# large numbers
    r'(cure|treat|prevent|cause)\s+\w+',  # health claims
    r'(increase|decrease|rise|fall)\s+by\s+\d+', # statistical claims
]

HEALTH_MISINFORMATION = [
    "cures cancer", "cures covid", "cures diabetes", "miracle cure",
    "doctors don't want", "big pharma hiding", "natural cure",
    "100% effective", "guaranteed to cure", "eliminates virus"
]

def claim_agent(text):
    """
    Extracts and evaluates factual claims from text.
    Flags potentially false or unverifiable claims.
    """
    text_lower = text.lower()
    sentences  = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 20]

    # Find sentences with claim indicators
    claim_sentences = []
    for sentence in sentences:
        for indicator in CLAIM_INDICATORS:
            if indicator in sentence.lower():
                claim_sentences.append(sentence.strip())
                break

    # Extract factual patterns
    factual_hits = []
    for pattern in FACTUAL_CLAIM_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        factual_hits.extend(matches)

    # Check health misinformation
    health_flags = [h for h in HEALTH_MISINFORMATION if h in text_lower]

    # Count unverifiable absolute claims
    absolute_words = ["always", "never", "everyone", "nobody", "all",
                      "proven", "fact", "definitely", "certainly", "100%"]
    absolute_found = [w for w in absolute_words if w in text_lower.split()]

    # Risk level
    risk_score = (
        len(health_flags) * 3 +
        len(absolute_found) * 0.5 +
        (1 if len(claim_sentences) > 3 else 0)
    )

    if risk_score >= 4:
        risk_level = "HIGH"
    elif risk_score >= 2:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "claim_count":       len(claim_sentences),
        "claims_found":      claim_sentences[:3],
        "factual_patterns":  list(set([str(f) for f in factual_hits]))[:5],
        "health_flags":      health_flags,
        "absolute_language": absolute_found,
        "claim_risk_level":  risk_level,
    }


# ─────────────────────────────────────────────
# COMBINED ANALYSIS
# ─────────────────────────────────────────────

def full_feature_analysis(text, title=""):
    """Run all 3 agents and return a combined analysis report."""
    emotion    = emotion_agent(text)
    credibility = credibility_agent(text, title)
    claims     = claim_agent(text)

    # Overall risk from features
    risk_points = 0
    if emotion["emotion_level"]      == "HIGH":   risk_points += 3
    elif emotion["emotion_level"]    == "MEDIUM":  risk_points += 1
    if credibility["credibility_level"] == "LOW":  risk_points += 3
    elif credibility["credibility_level"] == "MEDIUM": risk_points += 1
    if claims["claim_risk_level"]    == "HIGH":   risk_points += 3
    elif claims["claim_risk_level"]  == "MEDIUM":  risk_points += 1

    if risk_points >= 6:
        feature_verdict = "HIGH RISK"
    elif risk_points >= 3:
        feature_verdict = "MEDIUM RISK"
    else:
        feature_verdict = "LOW RISK"

    return {
        "emotion":          emotion,
        "credibility":      credibility,
        "claims":           claims,
        "feature_verdict":  feature_verdict,
        "risk_points":      risk_points,
    }


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_articles = [
        {
            "title": "SHOCKING: Government HIDING Alien Technology! Deep State Exposed!!!",
            "text":  "They don't want you to know the TRUTH! Whistleblowers confirm the deep state has been hiding alien technology for decades. Share this before it gets deleted! The mainstream media won't tell you this shocking secret. Wake up sheeple! 100% proven fact!!!"
        },
        {
            "title": "Federal Reserve raises interest rates by 0.25 percent",
            "text":  "The Federal Reserve announced Wednesday it would raise its benchmark interest rate by a quarter percentage point. According to Fed Chair Jerome Powell, the decision was unanimous. Reuters reports that markets had widely anticipated the move. The central bank said it remains committed to bringing inflation down to its 2 percent target."
        },
    ]

    for article in test_articles:
        print("\n" + "=" * 60)
        print(f"TITLE: {article['title']}")
        print("=" * 60)

        result = full_feature_analysis(article['text'], article['title'])

        print(f"\n🎭 EMOTION AGENT")
        print(f"   Level  : {result['emotion']['emotion_level']} (score: {result['emotion']['emotion_score']}/10)")
        print(f"   Flags  : {', '.join(result['emotion']['detected_phrases'][:5]) or 'None'}")

        print(f"\n📊 CREDIBILITY AGENT")
        print(f"   Level  : {result['credibility']['credibility_level']} (score: {result['credibility']['credibility_score']}/10)")
        print(f"   Sources: {', '.join(result['credibility']['credible_sources']) or 'None found'}")

        print(f"\n🔍 CLAIM AGENT")
        print(f"   Risk   : {result['claims']['claim_risk_level']}")
        print(f"   Flags  : {', '.join(result['claims']['health_flags']) or 'None'}")

        print(f"\n⚠️  FEATURE VERDICT: {result['feature_verdict']}")
