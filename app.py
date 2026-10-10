from flask import Flask, request, jsonify, render_template_string
import joblib, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocess import clean_text
from src.feature_agents import full_feature_analysis
from src.api_agents import api_analysis

app = Flask(__name__)
model      = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

TRUSTED_DOMAINS = [
    "reuters.com","bbc.com","apnews.com","theguardian.com","nytimes.com",
    "washingtonpost.com","bloomberg.com","npr.org","thehindu.com","ndtv.com",
    "timesofindia.com","abcnews.go.com","cbsnews.com","nbcnews.com",
    "factcheck.org","snopes.com","politifact.com","time.com","vox.com"
]

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fake News Detector</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
.header{background:#1e293b;padding:32px 20px;text-align:center;border-bottom:1px solid #334155}
.header h1{font-size:2.2rem;font-weight:700;color:#f8fafc}
.header h1 span{color:#ef4444}
.header p{color:#94a3b8;margin-top:8px}
.container{max-width:860px;margin:40px auto;padding:0 20px}
.card{background:#1e293b;border:1px solid #334155;border-radius:16px;padding:32px;margin-bottom:24px}
label{display:block;font-weight:600;color:#cbd5e1;margin-bottom:8px;margin-top:16px;font-size:.95rem}
input[type=text],textarea{width:100%;padding:14px 16px;background:#0f172a;border:1px solid #334155;border-radius:10px;color:#f1f5f9;font-size:.95rem;outline:none}
input:focus,textarea:focus{border-color:#6366f1}
textarea{resize:vertical;min-height:160px}
.btn{width:100%;padding:16px;background:#6366f1;color:#fff;border:none;border-radius:10px;font-size:1.05rem;font-weight:600;cursor:pointer;margin-top:20px}
.btn:hover{background:#4f46e5}
.btn:disabled{background:#334155;cursor:not-allowed}
.result-card{display:none}
.verdict-box{text-align:center;padding:32px;border-radius:12px;margin-bottom:24px}
.verdict-fake{background:rgba(239,68,68,.15);border:2px solid #ef4444}
.verdict-real{background:rgba(34,197,94,.15);border:2px solid #22c55e}
.verdict-label{font-size:2.8rem;font-weight:800;letter-spacing:3px}
.verdict-fake .verdict-label{color:#ef4444}
.verdict-real .verdict-label{color:#22c55e}
.confidence{font-size:1rem;color:#94a3b8;margin-top:8px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.acard{background:#0f172a;border:1px solid #334155;border-radius:12px;padding:20px;margin-bottom:16px}
.atitle{font-size:.78rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.aval{font-size:1.1rem;font-weight:700}
.high{color:#ef4444}.medium{color:#f59e0b}.low{color:#22c55e}
.verified{color:#22c55e}.unverified{color:#ef4444}.partial{color:#f59e0b}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.tag{padding:3px 9px;border-radius:20px;font-size:.73rem;background:rgba(239,68,68,.2);color:#fca5a5;border:1px solid rgba(239,68,68,.3)}
.sitem{padding:10px 14px;background:#1e293b;border-radius:8px;margin-bottom:8px;font-size:.85rem;border-left:3px solid #334155}
.sitem.trusted{border-left-color:#22c55e}
.sitem.flagged{border-left-color:#ef4444}
.loading{text-align:center;padding:40px;display:none}
.spinner{width:48px;height:48px;border:4px solid #334155;border-top-color:#6366f1;border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 16px}
@keyframes spin{to{transform:rotate(360deg)}}
.steps{display:flex;flex-direction:column;gap:8px;margin-top:16px;text-align:left;max-width:300px;margin-left:auto;margin-right:auto}
.st{font-size:.88rem;color:#64748b}
.st.active{color:#6366f1;font-weight:600}
.st.done{color:#22c55e}
.bar-wrap{height:6px;background:#334155;border-radius:4px;overflow:hidden;margin-top:8px}
.bar-fill{height:100%;border-radius:4px}
</style>
</head>
<body>
<div class="header">
  <h1>Fake News <span>Detector</span></h1>
  <p>Multi-agent AI — ML model + emotion analysis + real-time fact checking</p>
</div>
<div class="container">
  <div class="card">
    <label>Article Title</label>
    <input type="text" id="title" placeholder="Enter the news headline...">
    <label>Article Text</label>
    <textarea id="text" placeholder="Paste the full article text here..."></textarea>
    <button class="btn" onclick="analyze()" id="btn">Analyze Article</button>
  </div>
  <div class="loading" id="loading">
    <div class="spinner"></div>
    <p style="color:#94a3b8">Analyzing article...</p>
    <div class="steps">
      <div class="st" id="s1">Running ML model...</div>
      <div class="st" id="s2">Analyzing emotion and credibility...</div>
      <div class="st" id="s3">Checking Google Fact Check...</div>
      <div class="st" id="s4">Verifying with news sources...</div>
    </div>
  </div>
  <div class="result-card card" id="result">
    <div class="verdict-box" id="vbox">
      <div class="verdict-label" id="vlabel"></div>
      <div class="confidence" id="vconf"></div>
      <div style="margin-top:10px;font-size:.88rem;color:#94a3b8" id="vsub"></div>
    </div>
    <div class="grid">
      <div class="acard">
        <div class="atitle">ML Model</div>
        <div class="aval" id="mlv"></div>
        <div style="font-size:.8rem;color:#64748b;margin-top:4px" id="mls"></div>
        <div class="bar-wrap"><div class="bar-fill" id="mlb" style="width:0%"></div></div>
      </div>
      <div class="acard">
        <div class="atitle">Emotion Agent</div>
        <div class="aval" id="emv"></div>
        <div class="tags" id="emt"></div>
      </div>
      <div class="acard">
        <div class="atitle">Credibility Agent</div>
        <div class="aval" id="crv"></div>
        <div style="font-size:.8rem;color:#64748b;margin-top:4px" id="crs"></div>
      </div>
      <div class="acard">
        <div class="atitle">Claim Agent</div>
        <div class="aval" id="clv"></div>
        <div style="font-size:.8rem;color:#64748b;margin-top:4px" id="cls"></div>
      </div>
    </div>
    <div class="acard">
      <div class="atitle">Google Fact Check</div>
      <div class="aval" id="fcv"></div>
      <div id="fcs" style="margin-top:12px"></div>
    </div>
    <div class="acard">
      <div class="atitle">News Source Verification</div>
      <div class="aval" id="nvv"></div>
      <div id="nvs" style="margin-top:12px"></div>
    </div>
  </div>
</div>
<script>
function sleep(ms){return new Promise(r=>setTimeout(r,ms))}
function step(n){for(let i=1;i<=4;i++){let e=document.getElementById('s'+i);e.className=i<n?'st done':i===n?'st active':'st'}}
function gc(l){if(!l)return'';const u=l.toUpperCase();if(u.includes('HIGH')||u.includes('FALSE')||u.includes('UNVERIFIED'))return'high';if(u.includes('MEDIUM')||u.includes('PARTIAL')||u.includes('DISPUTED'))return'medium';return'low'}
async function analyze(){
  const title=document.getElementById('title').value.trim();
  const text=document.getElementById('text').value.trim();
  if(!title&&!text){alert('Please enter a title or article text.');return}
  document.getElementById('btn').disabled=true;
  document.getElementById('loading').style.display='block';
  document.getElementById('result').style.display='none';
  step(1);await sleep(500);step(2);await sleep(500);step(3);
  try{
    const res=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title,text})});
    step(4);await sleep(300);
    const d=await res.json();
    render(d);
  }catch(e){alert('Error: '+e.message)}
  document.getElementById('loading').style.display='none';
  document.getElementById('result').style.display='block';
  document.getElementById('btn').disabled=false;
}
function render(d){
  const fake=d.ml.label==='FAKE';
  const mc=fake?'#ef4444':'#22c55e';
  document.getElementById('vbox').className='verdict-box '+(fake?'verdict-fake':'verdict-real');
  document.getElementById('vlabel').textContent=fake?'FAKE NEWS':'REAL NEWS';
  document.getElementById('vconf').textContent=d.ml.confidence+'% confidence';
  document.getElementById('vsub').textContent='Emotion: '+d.features.emotion.emotion_level+' | Credibility: '+d.features.credibility.credibility_level+' | API: '+d.api.api_verdict;
  document.getElementById('mlv').textContent=d.ml.label;
  document.getElementById('mlv').style.color=mc;
  document.getElementById('mls').textContent='Fake: '+d.ml.fake_prob+'%  Real: '+d.ml.real_prob+'%';
  const b=document.getElementById('mlb');b.style.width=d.ml.fake_prob+'%';b.style.background=mc;
  const em=d.features.emotion;
  document.getElementById('emv').textContent=em.emotion_level+' ('+em.emotion_score+'/10)';
  document.getElementById('emv').className='aval '+gc(em.emotion_level);
  document.getElementById('emt').innerHTML=(em.detected_phrases||[]).slice(0,5).map(p=>'<span class="tag">'+p+'</span>').join('');
  const cr=d.features.credibility;
  document.getElementById('crv').textContent=cr.credibility_level+' ('+cr.credibility_score+'/10)';
  document.getElementById('crv').className='aval '+gc(cr.credibility_level);
  document.getElementById('crs').textContent='Words: '+cr.word_count+(cr.credible_sources.length?' | '+cr.credible_sources.join(', '):'');
  const cl=d.features.claims;
  document.getElementById('clv').textContent=cl.claim_risk_level+' risk';
  document.getElementById('clv').className='aval '+gc(cl.claim_risk_level);
  document.getElementById('cls').textContent=cl.claim_count+' claims'+(cl.health_flags.length?' | '+cl.health_flags.join(', '):'');
  const fc=d.api.fact_check;
  document.getElementById('fcv').textContent=fc.verdict||fc.status||'No results';
  document.getElementById('fcv').className='aval '+gc(fc.verdict||'');
  document.getElementById('fcs').innerHTML=(fc.results||[]).map(r=>'<div class="sitem '+(r.rating.toLowerCase().includes('false')?'flagged':'')+'"><strong>'+r.publisher+'</strong> — '+r.rating+'<br><small style="color:#64748b">'+r.claim_text.slice(0,90)+'</small></div>').join('')||'<div style="color:#64748b;font-size:.85rem">No matching fact-checks found</div>';
  const nv=d.api.news_verify;
  document.getElementById('nvv').textContent=nv.verification||nv.status||'No results';
  document.getElementById('nvv').className='aval '+(nv.verification==='VERIFIED'?'verified':nv.verification==='PARTIALLY VERIFIED'?'partial':'unverified');
  document.getElementById('nvs').innerHTML=(nv.articles||[]).map(a=>'<div class="sitem '+(a.trusted?'trusted':'')+'"><strong>'+a.source+'</strong>'+(a.trusted?' ✓':'')+'<br><small style="color:#64748b">'+a.title.slice(0,90)+'</small></div>').join('')||'<div style="color:#64748b;font-size:.85rem">No matching news sources found</div>';
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/analyze", methods=["POST"])
def analyze():
    data      = request.get_json()
    title     = data.get("title", "")
    text      = data.get("text", "")
    full_text = title + " " + text

    cleaned   = clean_text(full_text)
    tfidf     = vectorizer.transform([cleaned])
    proba     = model.predict_proba(tfidf)[0]
    ml_result = {
        "label":      "FAKE" if proba[1] > 0.5 else "REAL",
        "confidence": round(max(proba) * 100, 2),
        "fake_prob":  round(proba[1] * 100, 2),
        "real_prob":  round(proba[0] * 100, 2),
    }

    features = full_feature_analysis(text or title, title)
    api      = api_analysis(title, text)

    for article in api.get("news_verify", {}).get("articles", []):
        url    = article.get("url", "")
        domain = url.split("/")[2] if url.count("/") >= 2 else ""
        article["trusted"] = any(t in domain for t in TRUSTED_DOMAINS)

    return jsonify({"ml": ml_result, "features": features, "api": api})

if __name__ == "__main__":
    print("=" * 50)
    print("Fake News Detector running!")
    print("Open browser at: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, use_reloader=False)
