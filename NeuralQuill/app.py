"""
app.py  —  Shakespeare Generator Web Interface
Run: python3 app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
import os, sys, random
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from model import CharTokenizer, ShakespeareTransformer

TOK_PATH = os.path.join(BASE, 'tokenizer.pkl')
MDL_PATH = os.path.join(BASE, 'model.pkl')

# ── Load model ─────────────────────────────────────────────────
print("Loading model...")
tok   = CharTokenizer.load(TOK_PATH)
model = ShakespeareTransformer(
    vocab_size=tok.vocab_size,
    n_embd=128, n_head=4, n_layer=4, block_size=128
)
model.load(MDL_PATH)
print(f"Model ready! Vocab: {tok.vocab_size} chars, "
      f"Params: {model.P.count():,}")

# ── Flask ───────────────────────────────────────────────────────
app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Shakespeare Generator</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=IM+Fell+English:ital@0;1&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d0f0e;--surface:#131815;--surface2:#1a1f1c;
  --border:#1e2620;--border2:#273029;
  --gold:#c9a84c;--gold2:#e8c97a;--gold3:#8b6914;
  --green:#4a7c59;--green2:#5f9e72;
  --text:#ddd8c4;--text2:#8a8678;--text3:#50524a;
  --red:#8b3a3a;
}
html,body{height:100%;overflow:hidden}
body{background:var(--bg);color:var(--text);font-family:'JetBrains Mono',monospace;font-size:13px;display:flex;flex-direction:column;}

/* noise texture overlay */
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
  opacity:.4;
}

/* Header */
.header{
  flex-shrink:0;padding:18px 28px 16px;
  border-bottom:1px solid var(--border2);
  background:var(--surface);
  display:flex;align-items:center;justify-content:space-between;
  position:relative;z-index:10;
}
.logo{
  font-family:'IM Fell English',serif;font-style:italic;
  font-size:22px;color:var(--gold2);
  display:flex;align-items:center;gap:12px;
}
.logo-quill{font-size:20px;}
.logo-sub{
  font-size:9px;color:var(--text3);letter-spacing:.16em;
  text-transform:uppercase;display:block;margin-top:1px;
  font-family:'JetBrains Mono',monospace;font-style:normal;
}
.header-badges{display:flex;gap:8px;align-items:center;}
.badge{
  font-size:9px;letter-spacing:.1em;text-transform:uppercase;
  padding:3px 9px;border-radius:4px;border:1px solid;
}
.badge.gold{color:var(--gold);border-color:var(--gold3);background:var(--gold)11;}
.badge.green{color:var(--green2);border-color:var(--green);background:var(--green)11;}

/* Main layout */
.main{flex:1;display:flex;overflow:hidden;position:relative;z-index:1;}

/* Left panel — controls */
.controls{
  width:280px;flex-shrink:0;
  background:var(--surface);
  border-right:1px solid var(--border2);
  display:flex;flex-direction:column;
  overflow-y:auto;
}
.controls::-webkit-scrollbar{width:3px}
.controls::-webkit-scrollbar-thumb{background:var(--border2)}

.ctrl-section{padding:16px;border-bottom:1px solid var(--border);}
.ctrl-label{
  font-size:9px;color:var(--text3);letter-spacing:.14em;
  text-transform:uppercase;margin-bottom:10px;
  display:flex;align-items:center;gap:6px;
}
.ctrl-label::before{
  content:'';display:block;width:3px;height:9px;
  background:var(--gold);border-radius:2px;
}

/* Prompt input */
.prompt-wrap{position:relative;}
.prompt-input{
  width:100%;background:var(--surface2);
  border:1px solid var(--border2);border-radius:7px;
  padding:10px 12px;color:var(--text);
  font-family:'IM Fell English',serif;font-size:14px;
  resize:none;outline:none;line-height:1.6;
  transition:border-color .2s;
}
.prompt-input:focus{border-color:var(--gold)55;}
.prompt-input::placeholder{color:var(--text3);font-style:italic;}

.prompt-suggestions{
  display:flex;flex-wrap:wrap;gap:5px;margin-top:8px;
}
.sug-btn{
  font-size:10px;padding:3px 9px;
  background:var(--surface2);border:1px solid var(--border2);
  border-radius:4px;cursor:pointer;color:var(--text2);
  font-family:'IM Fell English',serif;font-style:italic;
  transition:all .15s;
}
.sug-btn:hover{border-color:var(--gold)55;color:var(--gold2);}

/* Slider controls */
.slider-row{margin-bottom:12px;}
.slider-head{
  display:flex;justify-content:space-between;
  font-size:11px;color:var(--text2);margin-bottom:6px;
}
.slider-val{color:var(--gold2);font-weight:500;}
.slider{
  width:100%;-webkit-appearance:none;appearance:none;
  height:3px;border-radius:2px;background:var(--border2);
  outline:none;cursor:pointer;
}
.slider::-webkit-slider-thumb{
  -webkit-appearance:none;width:14px;height:14px;
  border-radius:50%;background:var(--gold);cursor:pointer;
  border:2px solid var(--bg);
  box-shadow:0 0 6px var(--gold)66;
}
.slider-desc{font-size:9px;color:var(--text3);margin-top:4px;line-height:1.5;}

/* Generate button */
.gen-btn{
  width:100%;padding:11px;margin-top:4px;
  background:linear-gradient(135deg,var(--gold3),var(--gold));
  border:none;border-radius:8px;cursor:pointer;
  color:#1a1208;font-family:'IM Fell English',serif;
  font-size:15px;font-style:italic;
  display:flex;align-items:center;justify-content:center;gap:8px;
  transition:all .2s;font-weight:600;
}
.gen-btn:hover:not(:disabled){
  background:linear-gradient(135deg,var(--gold),var(--gold2));
  transform:translateY(-1px);
  box-shadow:0 4px 16px var(--gold)33;
}
.gen-btn:disabled{opacity:.5;cursor:not-allowed;transform:none;}

/* Quick prompts */
.quick-list{display:flex;flex-direction:column;gap:4px;}
.quick-item{
  padding:8px 10px;border-radius:6px;cursor:pointer;
  font-family:'IM Fell English',serif;font-style:italic;
  font-size:13px;color:var(--text2);
  border:1px solid transparent;
  transition:all .15s;
}
.quick-item:hover{
  background:var(--surface2);border-color:var(--border2);
  color:var(--gold2);
}

/* Stats */
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.stat-box{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:6px;padding:8px 10px;text-align:center;
}
.stat-n{
  font-family:'IM Fell English',serif;
  font-size:20px;color:var(--gold2);
}
.stat-l{font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.08em;}

/* Right panel — output */
.output-panel{
  flex:1;display:flex;flex-direction:column;overflow:hidden;
}
.output-header{
  padding:14px 22px;border-bottom:1px solid var(--border2);
  display:flex;align-items:center;justify-content:space-between;
  flex-shrink:0;
}
.output-title{
  font-family:'IM Fell English',serif;font-style:italic;
  font-size:15px;color:var(--text2);
}
.output-actions{display:flex;gap:6px;}
.act-btn{
  padding:5px 12px;border-radius:5px;cursor:pointer;
  font-size:10px;letter-spacing:.08em;text-transform:uppercase;
  border:1px solid var(--border2);background:none;color:var(--text3);
  transition:all .15s;font-family:'JetBrains Mono',monospace;
}
.act-btn:hover{background:var(--surface2);color:var(--text);}

.output-scroll{
  flex:1;overflow-y:auto;padding:32px 40px;
}
.output-scroll::-webkit-scrollbar{width:4px}
.output-scroll::-webkit-scrollbar-thumb{background:var(--border2)}

/* Empty state */
.empty{
  display:flex;flex-direction:column;align-items:center;
  justify-content:center;height:100%;gap:16px;
  color:var(--text3);text-align:center;
}
.empty-quill{font-size:52px;opacity:.2;}
.empty-title{
  font-family:'IM Fell English',serif;font-style:italic;
  font-size:24px;color:var(--text2);
}
.empty-sub{font-size:12px;line-height:1.7;max-width:320px;color:var(--text3);}

/* Generated text */
.gen-block{
  animation:fadeIn .5s ease;
  border:1px solid var(--border2);border-radius:10px;
  overflow:hidden;margin-bottom:20px;
}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.gen-meta{
  padding:10px 16px;background:var(--surface2);
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:10px;
  font-size:10px;color:var(--text3);
}
.gen-prompt-label{
  font-family:'IM Fell English',serif;font-style:italic;
  color:var(--gold);font-size:12px;
}
.gen-params{margin-left:auto;display:flex;gap:8px;}
.gen-param{
  background:var(--border);border-radius:3px;
  padding:1px 6px;font-size:9px;color:var(--text3);
}
.gen-text{
  padding:22px 24px;
  font-family:'IM Fell English',serif;
  font-size:15px;line-height:1.9;color:var(--text);
  white-space:pre-wrap;
  background:var(--surface);
}
.gen-text .prompt-part{color:var(--gold2);font-weight:500;}

/* Thinking */
.thinking{
  display:flex;align-items:center;gap:10px;
  padding:16px 24px;font-size:12px;color:var(--text3);
  font-family:'IM Fell English',serif;font-style:italic;
}
.pen{animation:write 1s ease infinite;}
@keyframes write{0%,100%{transform:rotate(-10deg)}50%{transform:rotate(10deg)}}
.tbar{height:2px;background:var(--border2);border-radius:2px;overflow:hidden;margin:0 0 16px;}
.tfill{height:100%;background:linear-gradient(90deg,var(--gold3),var(--gold));width:0%;transition:width .3s;}
</style>
</head>
<body>

<header class="header">
  <div class="logo">
    <span class="logo-quill">✒</span>
    <div>
      Shakespeare Generator
      <span class="logo-sub">Transformer · Built from Scratch · NumPy only</span>
    </div>
  </div>
  <div class="header-badges">
    <span class="badge gold">Character-level LM</span>
    <span class="badge green" id="paramBadge">Loading...</span>
  </div>
</header>

<div class="main">

  <!-- Controls -->
  <aside class="controls">

    <div class="ctrl-section">
      <div class="ctrl-label">Your Prompt</div>
      <div class="prompt-wrap">
        <textarea class="prompt-input" id="promptInput" rows="3"
          placeholder="HAMLET:&#10;To be, or not to be..."></textarea>
      </div>
      <div class="prompt-suggestions">
        <button class="sug-btn" onclick="setPrompt('HAMLET:\n')">HAMLET</button>
        <button class="sug-btn" onclick="setPrompt('ROMEO:\n')">ROMEO</button>
        <button class="sug-btn" onclick="setPrompt('MACBETH:\n')">MACBETH</button>
        <button class="sug-btn" onclick="setPrompt('JULIET:\n')">JULIET</button>
        <button class="sug-btn" onclick="setPrompt('KING LEAR:\n')">LEAR</button>
        <button class="sug-btn" onclick="setPrompt('To be, or')">To be</button>
        <button class="sug-btn" onclick="setPrompt('Friends, Romans,')">Romans</button>
      </div>
    </div>

    <div class="ctrl-section">
      <div class="ctrl-label">Generation Settings</div>

      <div class="slider-row">
        <div class="slider-head">
          <span>Length</span>
          <span class="slider-val" id="lenVal">400</span>
        </div>
        <input type="range" class="slider" id="lenSlider"
               min="100" max="1000" value="400" step="50"
               oninput="document.getElementById('lenVal').textContent=this.value"/>
        <div class="slider-desc">How many characters to generate</div>
      </div>

      <div class="slider-row">
        <div class="slider-head">
          <span>Temperature</span>
          <span class="slider-val" id="tempVal">0.80</span>
        </div>
        <input type="range" class="slider" id="tempSlider"
               min="0.3" max="1.5" value="0.8" step="0.05"
               oninput="document.getElementById('tempVal').textContent=parseFloat(this.value).toFixed(2)"/>
        <div class="slider-desc">Low = focused · High = creative/chaotic</div>
      </div>

      <div class="slider-row">
        <div class="slider-head">
          <span>Top-K</span>
          <span class="slider-val" id="topkVal">40</span>
        </div>
        <input type="range" class="slider" id="topkSlider"
               min="5" max="100" value="40" step="5"
               oninput="document.getElementById('topkVal').textContent=this.value"/>
        <div class="slider-desc">Candidates per step · Lower = more predictable</div>
      </div>

      <button class="gen-btn" id="genBtn" onclick="generate()">
        ✒ &nbsp;Generate
      </button>
    </div>

    <div class="ctrl-section">
      <div class="ctrl-label">Quick Prompts</div>
      <div class="quick-list">
        <div class="quick-item" onclick="quickGen('HAMLET:\n')">HAMLET: soliloquy</div>
        <div class="quick-item" onclick="quickGen('All the world\'s a stage,\n')">All the world's a stage</div>
        <div class="quick-item" onclick="quickGen('MACBETH:\nIs this a dagger')">Macbeth's dagger</div>
        <div class="quick-item" onclick="quickGen('ROMEO:\nBut, soft!')">Romeo at the window</div>
        <div class="quick-item" onclick="quickGen('PROSPERO:\nOur revels now')">Prospero's farewell</div>
        <div class="quick-item" onclick="quickGen('JULIET:\nO Romeo, Romeo!')">Juliet's balcony</div>
        <div class="quick-item" onclick="quickGen('FIRST WITCH:\nWhen shall we three')">The Three Witches</div>
      </div>
    </div>

    <div class="ctrl-section">
      <div class="ctrl-label">Stats</div>
      <div class="stat-grid">
        <div class="stat-box"><div class="stat-n" id="statGens">0</div><div class="stat-l">Generated</div></div>
        <div class="stat-box"><div class="stat-n" id="statChars">0</div><div class="stat-l">Chars out</div></div>
      </div>
    </div>

  </aside>

  <!-- Output -->
  <div class="output-panel">
    <div class="output-header">
      <div class="output-title">Generated Text</div>
      <div class="output-actions">
        <button class="act-btn" onclick="copyLast()">Copy</button>
        <button class="act-btn" onclick="clearAll()">Clear</button>
      </div>
    </div>
    <div class="tbar"><div class="tfill" id="tbar"></div></div>
    <div class="output-scroll" id="outputScroll">
      <div class="empty" id="emptyState">
        <div class="empty-quill">✒</div>
        <div class="empty-title">Awaiting your command</div>
        <div class="empty-sub">
          Enter a prompt or choose a quick prompt on the left.<br/>
          The model will continue in Shakespeare's style.
        </div>
      </div>
    </div>
  </div>

</div>

<script>
let gens=0, totalChars=0, lastText='';

// fetch model info
fetch('/info').then(r=>r.json()).then(d=>{
  document.getElementById('paramBadge').textContent =
    (d.params/1e6).toFixed(2)+'M params';
});

function setPrompt(p){
  document.getElementById('promptInput').value=p;
}

async function generate(){
  const prompt = document.getElementById('promptInput').value;
  if(!prompt.trim()) return;
  const length = parseInt(document.getElementById('lenSlider').value);
  const temp   = parseFloat(document.getElementById('tempSlider').value);
  const topk   = parseInt(document.getElementById('topkSlider').value);

  document.getElementById('emptyState')?.remove();
  document.getElementById('genBtn').disabled=true;

  // thinking indicator
  const thinking = document.createElement('div');
  thinking.className='thinking'; thinking.id='thinking';
  thinking.innerHTML='<span class="pen">✒</span> The model is writing...';
  document.getElementById('outputScroll').prepend(thinking);

  // progress bar animation
  let pct=0;
  const tbar=document.getElementById('tbar');
  const prog=setInterval(()=>{pct=Math.min(pct+2,90);tbar.style.width=pct+'%';},100);

  try{
    const r=await fetch('/generate',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt,length,temperature:temp,top_k:topk})
    });
    const d=await r.json();
    clearInterval(prog); tbar.style.width='100%';
    setTimeout(()=>{tbar.style.width='0%';},600);

    document.getElementById('thinking')?.remove();

    lastText=d.text;
    gens++; totalChars+=d.text.length;
    document.getElementById('statGens').textContent=gens;
    document.getElementById('statChars').textContent=
      totalChars>999?(totalChars/1000).toFixed(1)+'k':totalChars;

    const block=document.createElement('div');
    block.className='gen-block';
    const pLen=prompt.length;
    const promptPart=esc(d.text.slice(0,pLen));
    const genPart=esc(d.text.slice(pLen));
    block.innerHTML=`
      <div class="gen-meta">
        <span class="gen-prompt-label">${esc(prompt.trim().slice(0,30))}</span>
        <div class="gen-params">
          <span class="gen-param">len=${length}</span>
          <span class="gen-param">t=${temp}</span>
          <span class="gen-param">k=${topk}</span>
        </div>
      </div>
      <div class="gen-text"><span class="prompt-part">${promptPart}</span>${genPart}</div>`;
    document.getElementById('outputScroll').prepend(block);

  }catch(e){
    clearInterval(prog);
    document.getElementById('thinking')?.remove();
    const err=document.createElement('div');
    err.style.cssText='padding:16px;color:#8b3a3a;font-size:12px;';
    err.textContent='Error: '+e.message;
    document.getElementById('outputScroll').prepend(err);
  }

  document.getElementById('genBtn').disabled=false;
}

function quickGen(prompt){
  document.getElementById('promptInput').value=prompt;
  generate();
}

function copyLast(){
  if(lastText) navigator.clipboard.writeText(lastText)
    .then(()=>alert('Copied to clipboard!'));
}

function clearAll(){
  document.getElementById('outputScroll').innerHTML=
    '<div class="empty" id="emptyState">'+
    '<div class="empty-quill">✒</div>'+
    '<div class="empty-title">Awaiting your command</div>'+
    '<div class="empty-sub">Enter a prompt or choose a quick prompt on the left.</div>'+
    '</div>';
  gens=0;totalChars=0;lastText='';
  document.getElementById('statGens').textContent='0';
  document.getElementById('statChars').textContent='0';
}

function esc(s){return String(s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;')
  .replace(/>/g,'&gt;').replace(/\n/g,'<br/>');}
</script>
</body>
</html>"""

@app.route('/')
def index(): return render_template_string(HTML)

@app.route('/info')
def info():
    return jsonify({'params': model.P.count(), 'vocab': tok.vocab_size})

@app.route('/generate', methods=['POST'])
def gen():
    data  = request.get_json()
    prompt     = data.get('prompt', 'HAMLET:\n')
    length     = min(int(data.get('length', 400)), 1000)
    temperature= float(data.get('temperature', 0.8))
    top_k      = int(data.get('top_k', 40))

    ids  = tok.encode(prompt)[-128:]
    gen  = model.generate(ids, max_new=length,
                          temperature=temperature, top_k=top_k)
    text = tok.decode(gen)
    return jsonify({'text': text, 'chars': len(text)})

if __name__ == '__main__':
    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║  Shakespeare Generator                    ║")
    print("  ║  http://localhost:5000                    ║")
    print("  ╚══════════════════════════════════════════╝\n")
    app.run(debug=False, port=5000)
