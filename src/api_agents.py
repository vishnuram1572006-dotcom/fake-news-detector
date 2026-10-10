import requests

GOOGLE_FACT_CHECK_API_KEY = "AIzaSyD5p_zmeM65Vh8by5tu162rnmPGzxATPBs"
NEWS_API_KEY = "fc21df183d284327a642d657037ac292"

TRUSTED_DOMAINS = [
    "reuters.com", "bbc.com", "apnews.com", "theguardian.com",
    "nytimes.com", "washingtonpost.com", "bloomberg.com",
    "npr.org", "thehindu.com", "ndtv.com", "timesofindia.com"
]

def fact_check_agent(query, max_results=3):
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {"query": query[:200], "key": GOOGLE_FACT_CHECK_API_KEY, "pageSize": max_results}
        response = requests.get(url, params=params, timeout=10)
        claims = response.json().get("claims", [])
        if not claims:
            return {"status": "no_results", "results": [], "verdict": "UNVERIFIED"}
        results = []
        for claim in claims:
            reviews = claim.get("claimReview", [])
            rating = reviews[0].get("textualRating", "Unrated") if reviews else "Unrated"
            publisher = reviews[0].get("publisher", {}).get("name", "Unknown") if reviews else "Unknown"
            results.append({"claim_text": claim.get("text", "")[:150], "rating": rating, "publisher": publisher})
        ratings = [r["rating"].lower() for r in results]
        false_kw = ["false", "mostly false", "incorrect", "misleading", "fake", "pants on fire"]
        true_kw  = ["true", "mostly true", "correct", "accurate"]
        false_count = sum(1 for r in ratings if any(f in r for f in false_kw))
        true_count  = sum(1 for r in ratings if any(t in r for t in true_kw))
        verdict = "LIKELY FALSE" if false_count > true_count else ("LIKELY TRUE" if true_count > false_count else "DISPUTED")
        return {"status": "success", "verdict": verdict, "false_count": false_count, "true_count": true_count, "results": results[:3]}
    except Exception as e:
        return {"status": "error", "message": str(e), "results": [], "verdict": "UNKNOWN"}

def news_verification_agent(title, max_results=5):
    try:
        url = "https://newsapi.org/v2/everything"
        query = " ".join(title.split()[:8])
        params = {"q": query, "apiKey": NEWS_API_KEY, "pageSize": max_results, "language": "en", "sortBy": "relevancy"}
        response = requests.get(url, params=params, timeout=10)
        articles = response.json().get("articles", [])
        if not articles:
            return {"status": "no_results", "verification": "UNVERIFIED", "trusted_count": 0, "articles": []}
        results = []
        trusted_count = 0
        for article in articles:
            source_url = article.get("url", "")
            domain = source_url.split("/")[2] if source_url.count("/") >= 2 else ""
            is_trusted = any(t in domain for t in TRUSTED_DOMAINS)
            if is_trusted:
                trusted_count += 1
            results.append({"title": article.get("title", "")[:100], "source": article.get("source", {}).get("name", "Unknown"), "trusted": is_trusted})
        verification = "VERIFIED" if trusted_count >= 2 else ("PARTIALLY VERIFIED" if trusted_count == 1 else "UNVERIFIED")
        return {"status": "success", "verification": verification, "trusted_count": trusted_count, "articles": results[:3]}
    except Exception as e:
        return {"status": "error", "message": str(e), "verification": "UNKNOWN", "trusted_count": 0, "articles": []}

def api_analysis(title, text):
    print("  Checking Google Fact Check API...")
    fact_result = fact_check_agent(title + ". " + text[:200])
    print("  Checking NewsAPI...")
    news_result = news_verification_agent(title)
    api_risk = 0
    if fact_result.get("verdict") == "LIKELY FALSE": api_risk += 4
    elif fact_result.get("verdict") == "DISPUTED": api_risk += 2
    if news_result.get("verification") == "UNVERIFIED": api_risk += 2
    elif news_result.get("verification") == "PARTIALLY VERIFIED": api_risk += 1
    api_verdict = "HIGH RISK" if api_risk >= 4 else ("MEDIUM RISK" if api_risk >= 2 else "LOW RISK")
    return {"fact_check": fact_result, "news_verify": news_result, "api_verdict": api_verdict}

if __name__ == "__main__":
    articles = [
        {"title": "Federal Reserve raises interest rates", "text": "The Federal Reserve raised its benchmark interest rate to combat inflation."},
        {"title": "Drinking bleach cures COVID says study", "text": "Bleach can cure COVID-19 according to a shocking new study."},
    ]
    for a in articles:
        print("\n" + "=" * 60)
        print("TITLE:", a["title"])
        print("=" * 60)
        result = api_analysis(a["title"], a["text"])
        fc = result["fact_check"]
        nv = result["news_verify"]
        print("FACT CHECK:", fc.get("verdict", fc.get("status")))
        for r in fc.get("results", [])[:2]:
            print(" -", r["publisher"], "|", r["rating"], "|", r["claim_text"][:50])
        print("NEWS:", nv.get("verification"), "| trusted:", nv.get("trusted_count"))
        for r in nv.get("articles", [])[:2]:
            print(" -", r["source"], "| trusted:", r["trusted"], "|", r["title"][:50])
        print("API VERDICT:", result["api_verdict"])
