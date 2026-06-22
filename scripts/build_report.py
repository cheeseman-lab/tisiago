"""Build a self-contained, slide-based HTML report for tisiago (embeds the plot PNGs)."""

import base64
from pathlib import Path

OUT = Path("docs/tisiago_report.html")
OUT.parent.mkdir(parents=True, exist_ok=True)


def img64(p):
    p = Path(p)
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


PLOT_BEST = img64("autoresearch/plots/best_vs_floor.png")
PLOT_TRAJ = img64("autoresearch/plots/trajectories.png")

# --- codon-track diagram: same transcript framed as 3:1 vs codon-aware ---
N, START = 42, 9
COGNATE = {5, 9, 14, 21, 27, 33, 38}   # AUG/near-cognate positions (the start is one of them)
DECOY3 = {5, 21, 38}                    # the 3 matched decoys the 3:1 set uses as negatives
_c = lambda cls: f'<i class="c {cls}"></i>'  # noqa: E731
T_3TO1 = "".join(_c("pos" if i == START else "dec" if i in DECOY3 else "ign") for i in range(N))
T_CODON = "".join(_c("pos" if i == START else "neg" if i in COGNATE else "non") for i in range(N))

# --- shard chips (representative subset of 80, colored by partition) ---
def _shards():
    out = "".join(f'<span class="sh a6">{i}</span>' for i in range(6))
    out += '<span class="shx">&hellip;</span>' + "".join(f'<span class="sh a1">{i}</span>' for i in (48, 49))
    out += '<span class="shx">&hellip;</span>' + "".join(f'<span class="sh l4">{i}</span>' for i in (78, 79))
    return out

SHARDS = _shards()

# --- embedding matrix: rows = candidate codons, column blocks = keyed embeddings ---
_BLOCKS = [("ag", 7), ("ag", 7), ("ev", 9), ("oh", 2)]  # AG16k | AG131k | Evo2 | Kozak
EMB_ROWS = "".join(
    f'<div class="mrow"><span class="ml">codon&nbsp;{r}</span>'
    + "".join(f'<i class="m {cls}"></i>' for cls, n in _BLOCKS for _ in range(n))
    + "</div>"
    for r in range(6)
)

# --- worked example: one DNA region -> per-position embeddings -> 3 predictions ---
EX_SEQ = ["GCU", "ACG", "GAC", "AUG", "CUC", "GGA", "CUG", "UAC", "GCA", "AAA", "UUC", "GGC"]
EX_ROLE = {3: "pos", 6: "dec", 9: "non"}  # start / near-cognate decoy / non-cognate
EX_CODONS = "".join(f'<span class="cod {EX_ROLE.get(i, "f")}">{c}</span>' for i, c in enumerate(EX_SEQ))
EX_VECS = "".join(f'<span class="vec {EX_ROLE.get(i, "f")}"></span>' for i in range(len(EX_SEQ)))

SLIDES = f"""
<section class="slide title">
  <div class="kicker">fry-python-tools · translation initiation</div>
  <h1>tisiago</h1>
  <p class="lead">Probing whether <b>frozen</b> genome-language-model embeddings already encode
     <b>translation initiation</b> &mdash; a general <b>codon&nbsp;&rarr;&nbsp;P(start)</b> predictor.</p>
  <p class="sub">AlphaGenome&nbsp;+&nbsp;Evo2 · lightweight decoding heads · held-out chromosomes</p>
  <p class="status">A stepping stone, not a solved caller. Goal: any codon &rarr; a calibrated
     probability of initiation, including confidently <b>rejecting non-starts</b> (non-cognate&nbsp;&asymp;&nbsp;0).</p>
</section>

<section class="slide">
  <h2>How it works &mdash; frozen GLM, lightweight post-processing</h2>
  <div class="chain">
    <div class="step glm">DNA/RNA<br>sequence</div><div class="ar">&rarr;</div>
    <div class="step glm">frozen<br>genome&nbsp;LM</div><div class="ar">&rarr;</div>
    <div class="step glm">per-position<br>embeddings</div><div class="ar">&rarr;</div>
    <div class="step cpu">slice&nbsp;vector<br>@&nbsp;codon</div><div class="ar">&rarr;</div>
    <div class="step cpu">row-aligned<br>store [N,D]</div><div class="ar">&rarr;</div>
    <div class="step cpu">std&nbsp;&rarr;&nbsp;head<br>&rarr;&nbsp;calibrate</div><div class="ar">&rarr;</div>
    <div class="step out">P(start)</div>
  </div>
  <p class="legend"><span class="kg">&#9632; genome LM &mdash; GPU, frozen, run once</span>
     &nbsp;&nbsp;<span class="kc">&#9632; post-processing &mdash; CPU, cheap, iterated</span></p>
  <div class="two">
    <div class="card"><h3>What a genome LM gives us</h3><p>Self-supervised on massive genomes
       (next-token / masked), a GLM maps sequence &rarr; a <b>per-position embedding</b> encoding
       motif &amp; context &mdash; <b>no task labels</b>. We read a frozen hidden layer.
       <b>Evo2</b>: 1&nbsp;token/bp (base resolution). <b>AlphaGenome</b>: ~128&nbsp;bp-binned (regional).</p></div>
    <div class="card"><h3>What tisiago does to it</h3><p>Tile candidates into shared windows &rarr;
       one forward pass &rarr; <b>slice the vector at the candidate codon</b> (+offsets) &rarr; store
       fp16 <code>[N,D]</code> &rarr; <b>standardize &rarr; logistic/MLP head &rarr; isotonic
       calibration</b>. Every candidate is centered identically, so position carries no signal.</p></div>
  </div>
</section>

<section class="slide">
  <h2>Extraction &amp; sharding (GPU, run once)</h2>
  <div class="two">
    <div class="col">
      <div class="xstage">Manifest &mdash; up to 62.7M candidate codons</div>
      <div class="xarrow">shard by <code>crc32(transcript)&nbsp;%&nbsp;80</code><br>
        <span>&darr; keeps each transcript's tiles intact</span></div>
      <div class="shards">{SHARDS}</div>
      <div class="xlab">80 SLURM array tasks &mdash;
        <span class="kg">A6000</span> / <span class="ka1">A100</span> / <span class="kl">L40S</span></div>
      <div class="zoom"><div class="zt">inside one shard</div>
        <div class="zflow">
          <span class="zb">candidates</span><span class="za">&rarr;</span>
          <span class="zb">grid-snap<br>into tiles</span><span class="za">&rarr;</span>
          <span class="zb glm">1 forward pass<br>per tile</span><span class="za">&rarr;</span>
          <span class="zb">slice vector<br>@ each codon</span><span class="za">&rarr;</span>
          <span class="zb">.npz part</span>
        </div>
        <div class="znote">One forward pass per tile yields <b>every</b> candidate in it &rarr;
           GPU cost &prop; <b>tiles</b>, not candidates.</div>
      </div>
      <div class="xarrow"><code>store.py</code> scatters 80 parts<br>
        <span>&darr; into row-aligned arrays</span></div>
    </div>
    <div class="col">
      <h3>What an embedding store is</h3>
      <div class="emb">{EMB_ROWS}<div class="mdim">D = 5,632 &rarr; 19,620 dims (fp16)</div></div>
      <p>Each <b>row</b> = one codon's frozen context vector; <b>column blocks</b> = keyed
         encoders, <code>backend :: length :: layer :: offset</code>:</p>
      <ul class="keylist">
        <li><i class="kk ag"></i>AlphaGenome 16k &amp; 131k &mdash; 1536-d each</li>
        <li><i class="kk ev"></i>Evo2 blk28 &mdash; 4096-d &times; 4 offsets</li>
        <li><i class="kk oh"></i>Kozak one-hot &mdash; 164-d</li>
      </ul>
      <p class="cap">A head experiment = load a few <code>.npy</code>, <code>concat</code>, filter
         rows by split. <b>No GPU.</b></p>
    </div>
  </div>
</section>

<section class="slide">
  <h2>From DNA to a prediction &mdash; one region</h2>
  <div class="ex">
    <div class="exrow"><div class="exlab">genomic region<br><span>one transcript &rarr; one shard</span></div>
      <div class="seqstrip">{EX_CODONS}</div></div>
    <div class="exarrow">frozen GLM &mdash; <b>one forward pass</b> over the window &nbsp;&darr;</div>
    <div class="exrow"><div class="exlab">per-position<br><span>context embeddings</span></div>
      <div class="vecstrip">{EX_VECS}</div></div>
    <div class="exarrow">read the vector <b>at each candidate codon's location</b> (+offset) &nbsp;&darr;</div>
    <div class="preds">
      <div class="pr"><span class="vec pos"></span><span class="pa">&rarr;&nbsp;head&nbsp;&rarr;</span>
        <span class="pill g">AUG · P = 0.91</span><span class="prn">called start</span></div>
      <div class="pr"><span class="vec dec"></span><span class="pa">&rarr;&nbsp;head&nbsp;&rarr;</span>
        <span class="pill o">CUG · P = 0.08</span><span class="prn">uncalled near-cognate</span></div>
      <div class="pr"><span class="vec non"></span><span class="pa">&rarr;&nbsp;head&nbsp;&rarr;</span>
        <span class="pill r">AAA · P = 0.02</span><span class="prn">non-cognate (grounding)</span></div>
    </div>
  </div>
  <p class="cap"><b>Sharding</b> decides <i>where</i> the forward pass runs; the codon's
     <b>location</b> decides <i>which</i> vector the head reads. Same model, same region &mdash;
     three codons, three probabilities. (Illustrative values.)</p>
</section>

<section class="slide">
  <h2>Proof of concept &mdash; and how the 3:1 framing misleads</h2>
  <div class="two">
    <div class="col">
      <table>
        <tr><th>Curated 3:1 set</th><th>AUROC</th><th>win@64bp</th></tr>
        <tr><td>one-hot &plusmn;20bp (Kozak floor)</td><td>0.75</td><td>0.73</td></tr>
        <tr><td>AlphaGenome 16k</td><td>0.84</td><td>0.70</td></tr>
        <tr><td>Evo2 blk28</td><td>0.88</td><td>0.75</td></tr>
        <tr class="hl"><td>AlphaGenome + Evo2</td><td>0.90</td><td>0.82</td></tr>
      </table>
      <p class="cap">Real, but modest: honest lift is <b>0.75&rarr;0.90</b> over raw local sequence,
         not 0.5&rarr;0.90. Non-canonical starts hold up (0.886).</p>
    </div>
    <div class="col">
      <h3 class="fm">Failure modes of the 3:1 mode</h3>
      <ul class="fail">
        <li><b>AUROC is imbalance-blind</b> &mdash; it ranks; it never demands a confident &ldquo;no.&rdquo;</li>
        <li><b>Thin negatives</b> &mdash; only 3 matched near-cognate decoys; the head never sees
            the genome-scale negative distribution.</li>
        <li><b>Train-3:1 / test-230:1 mismatch</b> &mdash; at true imbalance, recall&nbsp;@&nbsp;&le;1&nbsp;FP/tx
            collapses to <b>~0.05</b> (AG-only).</li>
        <li><b>Regional inflation</b> &mdash; much of 0.90 is far-away decoys being easy; the honest
            base-resolution number is win@64 <b>0.82</b>.</li>
        <li><b>Grounding untested</b> &mdash; a ranking task never checks non-cognate&nbsp;&asymp;&nbsp;0.</li>
      </ul>
    </div>
  </div>
</section>

<section class="slide">
  <h2>Input framing &mdash; 3:1 vs codon-aware</h2>
  <p class="cap">Same transcript (each tick = one codon). Two very different questions.</p>
  <div class="track-block">
    <div class="tlab">3:1 &nbsp;<span>curated (PoC)</span></div>
    <div class="track">{T_3TO1}</div>
    <div class="tcap">1 start &middot; <b>3 matched near-cognate decoys</b> &middot; rest ignored
       &rarr; balanced, but blind to the genome.</div>
  </div>
  <div class="track-block">
    <div class="tlab">codon-aware &nbsp;<span>genome-wide (real task)</span></div>
    <div class="track">{T_CODON}</div>
    <div class="tcap"><b>every codon scored</b> &middot; 1 start vs ~570 candidates (<b>~230:1</b>)
       &middot; non-cognate held out as &asymp;0 grounding.</div>
  </div>
  <p class="legend2"><i class="c pos"></i>called start &nbsp; <i class="c dec"></i>3:1 decoy (neg)
     &nbsp; <i class="c neg"></i>cognate uncalled (neg) &nbsp; <i class="c non"></i>non-cognate (eval-only)
     &nbsp; <i class="c ign"></i>ignored</p>
  <p class="big2">&ldquo;Beat 3 decoys&rdquo; &nbsp;&ne;&nbsp; &ldquo;find the start among every codon.&rdquo;</p>
</section>

<section class="slide">
  <h2>Autoresearch &mdash; 4-objective fleet on the 3:1 set</h2>
  <div class="two imgrow">
    <div class="col"><img src="{PLOT_TRAJ}" alt="trajectories"/>
      <p class="cap">Four parallel loops, each climbing a different metric on <b>val</b>, reporting
         <b>test</b>. Fast early gains, plateau by ~exp 10&ndash;15.</p></div>
    <div class="col"><img src="{PLOT_BEST}" alt="best vs floor"/>
      <p class="cap">Codon control &asymp; chance; the jump is embeddings over the Kozak floor;
         autoresearch adds a <b>modest</b> lift (~+0.02&ndash;0.05). Logistic won every objective.</p></div>
  </div>
</section>

<section class="slide">
  <h2>Where this sits in the literature</h2>
  <div class="two">
    <div class="card"><h3>Empirical mapping</h3><p>Ribosome profiling, incl. initiation-specific
       protocols (LTM / harringtonine), maps TIS genome-wide &mdash; the label source here, but
       assay-dependent and noisy. <span class="ref">Ingolia 2009; Lee&nbsp;et&nbsp;al. 2012</span></p></div>
    <div class="card"><h3>Sequence predictors</h3><p>Task-trained CNN / transformer models of TIS
       from transcript sequence, learning Kozak-like context.
       <span class="ref">Kozak 1987; TITER 2017; TIS&nbsp;Transformer 2023</span></p></div>
    <div class="card"><h3>Genome foundation models</h3><p>General sequence encoders &mdash; <b>not</b>
       TIS-trained. <span class="ref">Evo2 (Arc, 2025); AlphaGenome (DeepMind, 2025)</span></p></div>
  </div>
  <p class="big2">tisiago's question: does a <b>general, frozen</b> genomic representation already
     contain the initiation signal &mdash; recoverable by a <b>lightweight</b> head, without
     task-specific training?</p>
  <p class="cap">(operon's specific literature search wasn't found on disk &mdash; this is the
     established landscape; point me to it and I'll swap its findings in.)</p>
</section>

<section class="slide">
  <h2>Status &mdash; established vs open</h2>
  <div class="two">
    <div class="card ok"><h3>Established</h3><ul>
      <li>The signal <b>exists</b> and is <b>linearly decodable</b> from frozen embeddings.</li>
      <li><b>Non-canonical</b> starts separate, not just canonical AUGs.</li>
      <li><b>Base resolution</b> is present (Evo2) and beats a raw-sequence floor.</li>
    </ul></div>
    <div class="card open"><h3>Open / unresolved</h3><ul>
      <li>Genome-wide <b>calling at true imbalance</b> &mdash; AG-only is a weak caller
          (an honest <b>negative</b> result, not buried).</li>
      <li>Whether <b>training on the real distribution &times; Evo2 base resolution</b> crosses
          from ranker to usable caller (in evaluation now).</li>
      <li><b>Grounding</b> non-cognate codons to &asymp;0.</li>
    </ul></div>
  </div>
  <p class="cap">Current step: train at the genome-scale imbalance (not 3:1), fold in Evo2, then
     autoresearch on the cached dense store. No overclaim &mdash; the calling problem is unsolved.</p>
</section>
"""

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>tisiago &mdash; report</title>
<style>
:root{ --ink:#1a2230; --mut:#5b6b82; --acc:#2563a8; --acc2:#0f9d8a; --warn:#c2410c; --line:#e2e8f0; --bg:#f7f9fc; }
*{box-sizing:border-box} html,body{margin:0;height:100%}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:var(--ink);background:var(--bg);overflow:hidden}
.deck{height:100vh;width:100vw;position:relative}
.slide{position:absolute;inset:0;display:none;flex-direction:column;justify-content:center;
  padding:5vh 7vw;animation:fade .35s ease}
.slide.active{display:flex}
@keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
h1{font-size:clamp(40px,7vw,84px);margin:.1em 0;letter-spacing:-1.5px}
h2{font-size:clamp(22px,3.2vw,38px);margin:0 0 .55em;color:var(--acc);letter-spacing:-.5px}
h3{font-size:17px;margin:0 0 .4em} h3.fm{color:var(--warn)}
.kicker{text-transform:uppercase;letter-spacing:3px;font-size:13px;color:var(--acc2);font-weight:700}
.lead{font-size:clamp(20px,2.6vw,30px);max-width:30ch;line-height:1.35} .lead b{color:var(--acc)}
.sub{font-size:18px;color:var(--mut);margin-top:.2em}
.status{margin-top:2em;font-size:14px;color:var(--mut);border-top:1px solid var(--line);
  padding-top:1em;max-width:70ch;line-height:1.5}
ul{font-size:clamp(14px,1.6vw,19px);line-height:1.5;max-width:62ch;padding-left:1.1em} li{margin:.38em 0}
b{color:var(--ink)} p{font-size:clamp(14px,1.6vw,19px);line-height:1.5;max-width:64ch}
.big2{font-size:clamp(18px,2.2vw,26px);font-weight:600;max-width:42ch;line-height:1.35;margin-top:.7em;color:var(--ink)}
.cap{font-size:clamp(12px,1.3vw,15px);color:var(--mut);max-width:80ch;margin-top:.8em}
.chain{display:flex;align-items:stretch;gap:6px;margin:.3em 0 .5em;flex-wrap:wrap;justify-content:center}
.step{flex:0 1 auto;text-align:center;border-radius:10px;padding:14px 12px;font-size:13px;font-weight:600;
  line-height:1.25;min-width:88px;border:1px solid var(--line);background:#fff}
.step.glm{border-color:#bcd4f0;background:#eef5ff;color:var(--acc)}
.step.cpu{border-color:#bfe7df;background:#ecfaf6;color:#0c7d6e}
.step.out{background:var(--ink);color:#fff;border-color:var(--ink)}
.ar{align-self:center;color:var(--mut);font-size:18px}
.legend{font-size:13px;color:var(--mut);text-align:center;margin:.2em 0 .8em}
.kg{color:var(--acc)} .kc{color:#0c7d6e}
.two{display:flex;gap:18px;flex-wrap:wrap;margin:.3em 0;align-items:flex-start}
.col{flex:1;min-width:300px} .card{flex:1;min-width:240px;background:#fff;border:1px solid var(--line);
  border-radius:12px;padding:16px} .card p{font-size:14px} .card ul{font-size:14px}
.card.ok{border-top:3px solid var(--acc2)} .card.open{border-top:3px solid var(--warn)}
.ref{display:block;color:var(--acc);font-size:12px;margin-top:.5em;font-style:italic}
table{border-collapse:collapse;width:100%;font-size:clamp(13px,1.5vw,18px);margin:.2em 0}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.5px}
tr.hl td{background:#eef5ff;font-weight:700}
img{max-width:100%;max-height:46vh;display:block;margin:.2em auto;border:1px solid var(--line);
  border-radius:10px;background:#fff;padding:5px}
.imgrow img{max-height:42vh}
.track-block{margin:.5em 0}
.tlab{font-size:15px;font-weight:700;color:var(--ink);margin-bottom:6px}
.tlab span{color:var(--mut);font-weight:500;font-size:12px}
.track{display:flex;gap:3px;flex-wrap:nowrap}
.c{width:15px;height:24px;border-radius:2px;background:#e5e9f0;flex:0 0 auto}
.c.pos{background:#16a34a} .c.dec{background:#ea8a2e} .c.neg{background:#e0628a}
.c.non{background:#cfd8e3} .c.ign{background:#eef1f6;opacity:.55}
.tcap{font-size:13px;color:var(--mut);margin-top:6px}
.legend2{font-size:12px;color:var(--mut);margin-top:1em;display:flex;align-items:center;gap:5px;flex-wrap:wrap}
.legend2 .c{width:13px;height:13px;margin-right:2px}
.xstage{background:var(--ink);color:#fff;border-radius:8px;padding:9px 14px;font-weight:600;font-size:14px;text-align:center}
.xarrow{text-align:center;font-size:12px;color:var(--mut);margin:6px 0} .xarrow span{display:block;font-size:11px}
.shards{display:flex;gap:4px;justify-content:center;align-items:center;flex-wrap:wrap;margin:2px 0}
.sh{width:24px;height:24px;border-radius:5px;font-size:11px;font-weight:700;color:#fff;display:flex;
  align-items:center;justify-content:center}
.sh.a6{background:#2563a8} .sh.a1{background:#0f9d8a} .sh.l4{background:#9333a8}
.shx{color:var(--mut);font-weight:700;padding:0 2px}
.xlab{text-align:center;font-size:12px;color:var(--mut);margin:5px 0 8px}
.ka1{color:#0f9d8a;font-weight:600} .kl{color:#9333a8;font-weight:600}
.zoom{border:1.5px dashed var(--line);border-radius:10px;padding:10px 12px;background:#fff}
.zt{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--mut);margin-bottom:7px}
.zflow{display:flex;align-items:center;gap:5px;flex-wrap:wrap;justify-content:center}
.zb{background:#f1f5fb;border:1px solid var(--line);border-radius:7px;padding:7px 8px;font-size:11px;
  font-weight:600;text-align:center;line-height:1.2} .zb.glm{background:#eef5ff;border-color:#bcd4f0;color:var(--acc)}
.za{color:var(--mut);font-size:14px} .znote{font-size:11.5px;color:var(--mut);margin-top:8px;text-align:center}
.emb{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px;margin:.2em 0 .6em}
.mrow{display:flex;align-items:center;gap:3px;margin:3px 0}
.ml{font-size:10px;color:var(--mut);width:54px;flex:0 0 auto;text-align:right;margin-right:5px}
.m{width:16px;height:13px;flex:0 0 auto;border-radius:1px}
.m.ag{background:#2563a8} .m.ev{background:#9333a8} .m.oh{background:#0f9d8a}
.mdim{font-size:11px;color:var(--mut);text-align:right;margin-top:7px}
.keylist{list-style:none;padding:0;font-size:14px} .keylist li{display:flex;align-items:center;margin:.3em 0}
.kk{width:14px;height:14px;border-radius:3px;margin-right:8px;flex:0 0 auto}
.kk.ag{background:#2563a8} .kk.ev{background:#9333a8} .kk.oh{background:#0f9d8a}
.ex{margin:.3em 0}
.exrow{display:flex;align-items:center;gap:16px;margin:5px 0}
.exlab{font-size:12px;color:var(--mut);width:120px;flex:0 0 auto;text-align:right;font-weight:700;line-height:1.3}
.exlab span{font-weight:400}
.seqstrip,.vecstrip{display:flex;gap:5px;flex-wrap:nowrap}
.cod{width:38px;text-align:center;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;
  font-weight:700;padding:7px 0;border-radius:5px;background:#eef1f6;color:var(--mut);letter-spacing:1px}
.cod.pos{background:#16a34a;color:#fff} .cod.dec{background:#ea8a2e;color:#fff}
.cod.non{background:#cfd8e3;color:#33425a} .cod.f{opacity:.55}
.vec{width:38px;height:30px;border-radius:4px;background:linear-gradient(#dbe4f0,#bccee4);flex:0 0 auto}
.vec.pos{background:linear-gradient(#34d399,#16a34a)} .vec.dec{background:linear-gradient(#fbbf77,#ea8a2e)}
.vec.non{background:linear-gradient(#dde4ee,#cfd8e3)} .vec.f{opacity:.4}
.exarrow{text-align:center;font-size:12.5px;color:var(--mut);margin:8px 0}
.preds{margin-top:6px} .pr{display:flex;align-items:center;gap:11px;margin:8px 0}
.pr .vec{width:24px;height:26px} .pa{font-size:12px;color:var(--mut)}
.pill{font-size:13px;font-weight:700;padding:6px 13px;border-radius:16px;color:#fff;white-space:nowrap}
.pill.g{background:#16a34a} .pill.o{background:#ea8a2e} .pill.r{background:#b04a6a}
.prn{font-size:12px;color:var(--mut)}
.hud{position:fixed;bottom:18px;left:0;right:0;display:flex;align-items:center;justify-content:center;gap:16px;z-index:10}
.dots{display:flex;gap:7px} .dot{width:8px;height:8px;border-radius:50%;background:#cbd5e1;cursor:pointer}
.dot.on{background:var(--acc);transform:scale(1.25)}
.pg{position:fixed;bottom:16px;right:24px;font-size:12px;color:var(--mut);letter-spacing:1px}
.brand{position:fixed;top:18px;left:24px;font-size:12px;color:var(--mut);letter-spacing:2px;text-transform:uppercase}
.nav{background:#fff;border:1px solid var(--line);border-radius:20px;width:34px;height:34px;cursor:pointer;
  font-size:16px;line-height:32px;text-align:center;user-select:none}
</style></head><body>
<div class="brand">tisiago</div>
<div class="deck">__SLIDES__</div>
<div class="hud"><div class="nav" onclick="go(cur-1)">&#8249;</div>
  <div class="dots" id="dots"></div><div class="nav" onclick="go(cur+1)">&#8250;</div></div>
<div class="pg" id="pg"></div>
<script>
const slides=[...document.querySelectorAll('.slide')];let cur=0;const dots=document.getElementById('dots');
slides.forEach((_,i)=>{const d=document.createElement('div');d.className='dot';d.onclick=()=>go(i);dots.appendChild(d);});
function go(i){cur=Math.max(0,Math.min(slides.length-1,i));
  slides.forEach((s,j)=>s.classList.toggle('active',j===cur));
  [...dots.children].forEach((d,j)=>d.classList.toggle('on',j===cur));
  document.getElementById('pg').textContent=(cur+1)+' / '+slides.length;}
document.addEventListener('keydown',e=>{if(['ArrowRight',' ','PageDown'].includes(e.key)){e.preventDefault();go(cur+1)}
  if(['ArrowLeft','PageUp'].includes(e.key))go(cur-1);if(e.key==='Home')go(0);if(e.key==='End')go(slides.length-1);});
go(0);
</script></body></html>"""

OUT.write_text(HTML.replace("__SLIDES__", SLIDES))
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {SLIDES.count('<section')} slides)")
