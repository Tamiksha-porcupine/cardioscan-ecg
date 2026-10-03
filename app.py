"""
app.py - ECG Analysis Web Application with XAI (Grad-CAM + SHAP)
Run with: python app.py
Then open: http://localhost:5000
"""

import os, sys, json, base64, io
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify, render_template_string

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
NORM_MEAN  = [0.485, 0.456, 0.406]
NORM_STD   = [0.229, 0.224, 0.225]

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_DESCRIPTIONS = {
    "NORM": "Normal ECG",
    "MI"  : "Myocardial Infarction",
    "STTC": "ST/T-Wave Change",
    "CD"  : "Conduction Disturbance",
    "HYP" : "Hypertrophy",
}
CLASS_COLORS = {
    "NORM": "#2D9B6F",
    "MI"  : "#D94F3D",
    "STTC": "#E07B2A",
    "CD"  : "#5B7FD4",
    "HYP" : "#9B59B6",
}
CLASS_ADVICE = {
    "NORM": "No abnormalities detected. Maintain regular cardiac checkups as recommended by your physician.",
    "MI"  : "Signs consistent with myocardial infarction detected. Please seek immediate medical attention.",
    "STTC": "ST/T-wave changes detected. Consult a cardiologist for further evaluation.",
    "CD"  : "Conduction disturbance detected. A follow-up with a cardiac specialist is recommended.",
    "HYP" : "Signs of hypertrophy detected. Please consult your physician for further assessment.",
}

WEIGHTS_PATH = Path(__file__).parent / "models" / "ecg_best.pth"

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

def load_model():
    from model_OPTIMIZED import build_model
    model = build_model(weights_path=str(WEIGHTS_PATH), eval_mode=True)
    model.to(DEVICE)
    model.eval()
    return model

print("Loading model...")
model = load_model()
print("Model ready.")

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks      = []

    def _register(self):
        target = self.model.encoder.eff.features[-1]
        def fwd(m, inp, out): self.activations = out.detach()
        def bwd(m, gi, go):   self.gradients   = go[0].detach()
        self._hooks.append(target.register_forward_hook(fwd))
        self._hooks.append(target.register_full_backward_hook(bwd))

    def _remove(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()

    def generate(self, tensor, class_idx):
        self._register()
        self.model.zero_grad()
        tensor = tensor.clone().requires_grad_(True)
        _, logits = self.model(tensor)
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        self._remove()
        self.model.zero_grad()
        return cam

def gradcam_overlay(pil_img, heatmap):
    orig     = np.array(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")) / 255.0
    heat_rgb = cm.get_cmap("jet")(heatmap)[:,:,:3]
    blended  = np.clip(0.55*orig + 0.45*heat_rgb, 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(orig);    axes[0].axis("off"); axes[0].set_title("Original ECG", fontsize=11)
    axes[1].imshow(blended); axes[1].axis("off"); axes[1].set_title("Grad-CAM Heatmap", fontsize=11)
    sm   = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0,1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Attention intensity", fontsize=9)
    cbar.set_ticks([0, 0.5, 1]); cbar.set_ticklabels(["Low","Medium","High"])
    fig.suptitle("Grad-CAM: regions that influenced the prediction", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def compute_shap_scores(tensor, pred_idx, n_samples=20):
    baseline  = torch.zeros_like(tensor)
    alphas    = torch.linspace(0, 1, n_samples).to(DEVICE)
    grads_sum = torch.zeros_like(tensor)
    for alpha in alphas:
        interp = (baseline + alpha*(tensor - baseline)).requires_grad_(True)
        _, logits = model(interp)
        logits[0, pred_idx].backward()
        grads_sum += interp.grad.detach()
    ig = (tensor - baseline) * grads_sum / n_samples
    return ig.squeeze().abs().mean(dim=(1,2)).cpu().numpy()

def shap_bar_chart(scores, pred_class):
    channels = ["Red channel\n(intensity/contrast)", "Green channel\n(waveform shape)", "Blue channel\n(background/grid)"]
    colors   = ["#E74C3C","#27AE60","#2980B9"]
    norm_s   = scores / (scores.sum() + 1e-8) * 100
    fig, ax  = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(channels, norm_s, color=colors, height=0.5, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, norm_s):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f"{val:.1f}%", va="center", fontsize=10)
    ax.set_xlabel("Relative attribution (%)", fontsize=10)
    ax.set_title(f"Feature attribution for predicted class: {pred_class}", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(norm_s)*1.2)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    fig.patch.set_facecolor("#FAFAFA"); ax.set_facecolor("#FAFAFA")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def predict_from_pil(img):
    tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        waveform, logits = model(tensor)
    probs    = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_cls = CLASS_NAMES[pred_idx]
    conf     = round(float(probs[pred_idx])*100, 1)

    # Waveform plot
    wave = waveform.squeeze().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(wave, color="#1B4F8A", linewidth=1.2)
    ax.fill_between(range(len(wave)), wave, alpha=0.08, color="#1B4F8A")
    ax.set_xlabel("Sample", fontsize=11, color="#555")
    ax.set_ylabel("Amplitude", fontsize=11, color="#555")
    ax.set_title("Predicted ECG Waveform (next 500 samples)", fontsize=12, pad=12)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    fig.patch.set_facecolor("#FAFAFA"); ax.set_facecolor("#FAFAFA")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    waveform_b64 = base64.b64encode(buf.getvalue()).decode()

    # Grad-CAM
    gcam        = GradCAM(model)
    heatmap     = gcam.generate(tensor.clone(), pred_idx)
    gradcam_b64 = gradcam_overlay(img, heatmap)

    # SHAP
    shap_scores = compute_shap_scores(tensor.clone(), pred_idx, n_samples=20)
    shap_b64    = shap_bar_chart(shap_scores, pred_cls)

    # Plain English explanations (built in Python to avoid JS string issues)
    if conf > 65:
        gcam_explain = (
            "The red and orange areas show where the AI focused most attention. "
            "These are the regions of your ECG that most strongly influenced the "
            + pred_cls + " diagnosis. Blue and purple areas were largely ignored. "
            "Think of it like a highlights reel showing which parts of the ECG the model read most carefully, "
            "similar to how a cardiologist would focus on specific waves and segments."
        )
    else:
        gcam_explain = (
            "The heatmap is fairly spread out, which reflects the model uncertainty. "
            "It found relevant signals in multiple regions rather than one clear area. "
            "This is why the confidence is lower (" + str(conf) + "%). "
            "The model detected overlapping features across the ECG rather than one dominant abnormality."
        )

    sorted_probs = sorted(probs, reverse=True)
    top1_pct  = round(float(sorted_probs[0])*100, 1)
    top2_pct  = round(float(sorted_probs[1])*100, 1)
    norm_shap = shap_scores / (shap_scores.sum() + 1e-8) * 100
    top_ch_idx = int(np.argmax(norm_shap))
    ch_names   = ["Red (intensity/contrast)", "Green (waveform shape)", "Blue (background/grid)"]
    shap_explain = (
        "The three bars show which colour channel the AI relied on most. "
        "Red captures brightness and contrast of the ECG waves. "
        "Green captures the actual shape and curves of the waveform. "
        "Blue captures the grid lines and beat spacing. "
        "For your result (" + pred_cls + "), the most influential channel was "
        + ch_names[top_ch_idx] + " at " + str(round(float(norm_shap[top_ch_idx]),1)) + "%. "
        "The three channels were " + ("evenly weighted, meaning no single feature dominated." if max(norm_shap) < 40 else "clearly dominated by one feature type.")
    )

    return {
        "predicted_class" : pred_cls,
        "description"     : CLASS_DESCRIPTIONS[pred_cls],
        "confidence"      : conf,
        "color"           : CLASS_COLORS[pred_cls],
        "advice"          : CLASS_ADVICE[pred_cls],
        "probabilities"   : {cls: round(float(p)*100,1) for cls,p in zip(CLASS_NAMES, probs)},
        "colors"          : CLASS_COLORS,
        "waveform_img"    : waveform_b64,
        "gradcam_img"     : gradcam_b64,
        "shap_img"        : shap_b64,
        "gcam_explain"    : gcam_explain,
        "shap_explain"    : shap_explain,
    }

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>CardioScan - ECG Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --ink: #1A1E2E; --ink-soft: #4A5068; --ink-muted: #8A90A8;
    --surface: #F5F6FA; --white: #FFFFFF; --border: #E2E5F0;
    --blue: #1B4F8A; --blue-lt: #EBF1FA; --radius: 14px;
    --shadow: 0 2px 16px rgba(26,30,46,0.08);
  }
  body { font-family:'DM Sans',sans-serif; background:var(--surface); color:var(--ink); min-height:100vh; }
  header { background:var(--white); border-bottom:1px solid var(--border); padding:0 2rem; display:flex; align-items:center; justify-content:space-between; height:64px; position:sticky; top:0; z-index:100; }
  .logo { display:flex; align-items:center; gap:10px; text-decoration:none; }
  .logo-icon { width:36px; height:36px; background:var(--blue); border-radius:9px; display:flex; align-items:center; justify-content:center; }
  .logo-name { font-family:'DM Serif Display',serif; font-size:1.3rem; color:var(--ink); }
  .logo-name span { color:var(--blue); }
  .badge { font-size:0.7rem; font-weight:500; background:var(--blue-lt); color:var(--blue); padding:3px 10px; border-radius:20px; }
  .hero { background:var(--white); border-bottom:1px solid var(--border); padding:3.5rem 2rem 3rem; text-align:center; }
  .hero-eyebrow { font-size:0.78rem; font-weight:500; color:var(--blue); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:1rem; }
  .hero h1 { font-family:'DM Serif Display',serif; font-size:clamp(2rem,5vw,3.2rem); color:var(--ink); line-height:1.15; letter-spacing:-0.02em; margin-bottom:1rem; max-width:680px; margin-left:auto; margin-right:auto; }
  .hero p { color:var(--ink-soft); font-size:1.05rem; max-width:520px; margin:0 auto; line-height:1.65; font-weight:300; }
  .pulse-bar { height:48px; overflow:hidden; margin:2rem 0 0; }
  .pulse-bar svg { width:100%; height:48px; opacity:0.18; }
  .main { max-width:900px; margin:0 auto; padding:2.5rem 1.5rem 4rem; }
  .card { background:var(--white); border:1px solid var(--border); border-radius:var(--radius); box-shadow:var(--shadow); overflow:hidden; }
  .card-header { padding:1.25rem 1.75rem; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:10px; }
  .card-header h2 { font-size:1rem; font-weight:600; color:var(--ink); }
  .step-dot { width:26px; height:26px; background:var(--blue); color:#fff; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:600; flex-shrink:0; }
  .drop-zone { margin:1.75rem; border:2px dashed var(--border); border-radius:10px; padding:3rem 1.5rem; text-align:center; cursor:pointer; transition:border-color 0.2s,background 0.2s; position:relative; }
  .drop-zone:hover,.drop-zone.drag-over { border-color:var(--blue); background:var(--blue-lt); }
  .drop-zone input[type="file"] { position:absolute; inset:0; opacity:0; cursor:pointer; width:100%; height:100%; }
  .drop-icon { width:52px; height:52px; background:var(--blue-lt); border-radius:12px; display:flex; align-items:center; justify-content:center; margin:0 auto 1.25rem; }
  .drop-icon svg { width:26px; height:26px; color:var(--blue); }
  .drop-zone h3 { font-size:1rem; font-weight:600; color:var(--ink); margin-bottom:0.4rem; }
  .drop-zone p { font-size:0.85rem; color:var(--ink-muted); }
  .drop-zone p strong { color:var(--blue); font-weight:500; }
  #preview-wrap { margin:0 1.75rem 1.75rem; display:none; gap:1.25rem; align-items:flex-start; }
  #preview-wrap.visible { display:flex; }
  #preview-img { width:160px; height:120px; object-fit:cover; border-radius:8px; border:1px solid var(--border); flex-shrink:0; }
  .preview-meta .fname { font-weight:500; font-size:0.9rem; color:var(--ink); margin-bottom:0.3rem; word-break:break-all; }
  .preview-meta .fsize { font-size:0.8rem; color:var(--ink-muted); }
  .btn-analyze { display:block; width:calc(100% - 3.5rem); margin:0 1.75rem 1.75rem; padding:0.9rem; background:var(--blue); color:#fff; border:none; border-radius:9px; font-family:'DM Sans',sans-serif; font-size:1rem; font-weight:600; cursor:pointer; transition:background 0.18s; }
  .btn-analyze:hover { background:#163E6E; }
  .btn-analyze:disabled { background:var(--ink-muted); cursor:not-allowed; }
  #loading { display:none; margin:1.75rem; gap:12px; align-items:center; }
  #loading.visible { display:flex; }
  .spinner { width:22px; height:22px; border:3px solid var(--border); border-top-color:var(--blue); border-radius:50%; animation:spin 0.75s linear infinite; flex-shrink:0; }
  @keyframes spin { to { transform:rotate(360deg); } }
  #loading span { color:var(--ink-soft); font-size:0.9rem; }
  #result { display:none; margin-top:1.75rem; }
  #result.visible { display:block; }
  .result-header { padding:1.5rem 1.75rem; display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; }
  .diagnosis-block { display:flex; align-items:center; gap:14px; }
  .diagnosis-pill { padding:0.45rem 1.1rem; border-radius:30px; font-weight:700; font-size:1rem; letter-spacing:0.04em; color:#fff; }
  .diagnosis-label { font-family:'DM Serif Display',serif; font-size:1.5rem; color:var(--ink); line-height:1.2; }
  .diagnosis-sub { font-size:0.85rem; color:var(--ink-muted); margin-top:2px; }
  .confidence-ring { text-align:center; flex-shrink:0; }
  .confidence-ring svg { width:80px; height:80px; }
  .confidence-ring .val { font-size:1.1rem; font-weight:700; fill:var(--ink); font-family:'DM Sans',sans-serif; }
  .confidence-ring .lbl { font-size:0.55rem; fill:var(--ink-muted); font-family:'DM Sans',sans-serif; }
  .advice-banner { margin:0 1.75rem 1.25rem; padding:0.9rem 1.1rem; border-radius:9px; font-size:0.88rem; line-height:1.55; border-left:4px solid; }
  .prob-section { padding:1.5rem 1.75rem; border-top:1px solid var(--border); }
  .section-title { font-size:0.8rem; font-weight:600; color:var(--ink-muted); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:1rem; }
  .prob-row { display:flex; align-items:center; gap:10px; margin-bottom:0.65rem; }
  .prob-label { width:48px; font-size:0.82rem; font-weight:600; color:var(--ink-soft); flex-shrink:0; }
  .prob-track { flex:1; height:8px; background:var(--surface); border-radius:4px; overflow:hidden; }
  .prob-fill { height:100%; border-radius:4px; transition:width 0.8s cubic-bezier(0.4,0,0.2,1); }
  .prob-val { width:44px; font-size:0.82rem; color:var(--ink-muted); text-align:right; flex-shrink:0; }
  .wave-section { padding:1.5rem 1.75rem; border-top:1px solid var(--border); }
  .wave-section img { width:100%; border-radius:8px; border:1px solid var(--border); }
  .xai-section { border-top:1px solid var(--border); }
  .xai-toggle { width:100%; padding:1.25rem 1.75rem; display:flex; align-items:center; justify-content:space-between; background:none; border:none; cursor:pointer; font-family:'DM Sans',sans-serif; font-size:0.95rem; font-weight:600; color:var(--ink); text-align:left; transition:background 0.15s; }
  .xai-toggle:hover { background:var(--surface); }
  .xai-toggle-left { display:flex; align-items:center; gap:10px; }
  .xai-chip { font-size:0.68rem; font-weight:600; letter-spacing:0.06em; background:#F0F4FF; color:var(--blue); padding:3px 9px; border-radius:20px; text-transform:uppercase; }
  .xai-arrow { width:20px; height:20px; transition:transform 0.25s; color:var(--ink-muted); flex-shrink:0; }
  .xai-arrow.open { transform:rotate(180deg); }
  .xai-body { display:none; padding:0 1.75rem 1.75rem; }
  .xai-body.open { display:block; }
  .xai-intro { font-size:0.85rem; color:var(--ink-soft); line-height:1.6; background:var(--surface); border-radius:9px; padding:0.9rem 1.1rem; margin-bottom:1.5rem; border-left:3px solid var(--blue); }
  .xai-block { margin-bottom:1.75rem; }
  .xai-block-title { font-size:0.8rem; font-weight:600; color:var(--ink-muted); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.75rem; display:flex; align-items:center; gap:8px; }
  .xai-block-title .dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
  .xai-block img { width:100%; border-radius:8px; border:1px solid var(--border); }
  .xai-explain { margin-top:0.9rem; background:#F8F9FF; border:1px solid #DDE3F5; border-radius:9px; padding:1rem 1.15rem; }
  .xai-explain-title { font-size:0.75rem; font-weight:700; color:var(--blue); text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.55rem; }
  .xai-explain p { font-size:0.87rem; color:var(--ink-soft); line-height:1.65; }
  .disclaimer { margin-top:2rem; padding:1rem 1.25rem; background:#FFF8E6; border:1px solid #F0DFA0; border-radius:9px; font-size:0.8rem; color:#7A6520; line-height:1.55; display:flex; gap:10px; align-items:flex-start; }
  @media(max-width:600px){
    header { padding:0 1rem; }
    .hero { padding:2.5rem 1rem 2rem; }
    .main { padding:1.5rem 1rem 3rem; }
    .drop-zone { margin:1rem; padding:2rem 1rem; }
    .btn-analyze { width:calc(100% - 2rem); margin:0 1rem 1rem; }
    #preview-wrap { margin:0 1rem 1rem; flex-direction:column; }
    #preview-img { width:100%; height:160px; }
    .result-header { flex-direction:column; align-items:flex-start; }
  }
</style>
</head>
<body>
<header>
  <a class="logo" href="#">
    <div class="logo-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round">
        <polyline points="2,12 5,12 7,6 9,18 11,10 13,14 15,12 22,12"/>
      </svg>
    </div>
    <span class="logo-name">Cardio<span>Scan</span></span>
  </a>
  <span class="badge">AI-Powered · XAI</span>
</header>

<div class="hero">
  <p class="hero-eyebrow">ECG Analysis Platform</p>
  <h1>Instant cardiac insights from your ECG report</h1>
  <p>Upload any ECG image and our hybrid AI model returns a diagnosis with full explainability.</p>
  <div class="pulse-bar">
    <svg viewBox="0 0 1200 48" preserveAspectRatio="none">
      <polyline points="0,24 120,24 160,4 200,44 240,14 280,34 320,24 400,24 440,8 480,40 520,18 560,30 600,24 680,6 720,42 760,16 800,32 840,24 920,24 960,10 1000,38 1040,20 1080,28 1120,24 1200,24"
        fill="none" stroke="#1B4F8A" stroke-width="2.5"/>
    </svg>
  </div>
</div>

<main class="main">
  <div class="card">
    <div class="card-header">
      <div class="step-dot">1</div>
      <h2>Upload your ECG image</h2>
    </div>
    <div class="drop-zone" id="dropZone">
      <input type="file" id="fileInput" accept="image/*"/>
      <div class="drop-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>
      <h3>Drop your ECG image here</h3>
      <p>or <strong>click to browse</strong> &mdash; PNG, JPG, JPEG accepted</p>
    </div>
    <div id="preview-wrap">
      <img id="preview-img" src="" alt="Preview"/>
      <div class="preview-meta">
        <div class="fname" id="fname"></div>
        <div class="fsize" id="fsize"></div>
      </div>
    </div>
    <button class="btn-analyze" id="analyzeBtn" disabled onclick="analyze()">Analyze ECG</button>
    <div id="loading">
      <div class="spinner"></div>
      <span>Analyzing ECG and computing XAI &mdash; may take 1-2 min on CPU...</span>
    </div>
  </div>

  <div class="card" id="result">
    <div class="card-header">
      <div class="step-dot">2</div>
      <h2>Analysis result</h2>
    </div>
    <div class="result-header" id="resultHeader"></div>
    <div id="adviceBanner" class="advice-banner"></div>
    <div class="prob-section">
      <div class="section-title">Class probabilities</div>
      <div id="probBars"></div>
    </div>
    <div class="wave-section">
      <div class="section-title" style="margin-bottom:0.75rem">Predicted ECG waveform</div>
      <img id="waveImg" src="" alt="Predicted waveform"/>
    </div>
    <div class="xai-section">
      <button class="xai-toggle" onclick="toggleXAI()">
        <div class="xai-toggle-left">
          <span>Explainability (XAI)</span>
          <span class="xai-chip">Grad-CAM + SHAP</span>
        </div>
        <svg class="xai-arrow" id="xaiArrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
          <polyline points="6 9 12 15 18 9"/>
        </svg>
      </button>
      <div class="xai-body" id="xaiBody">
        <div class="xai-intro">
          These visualisations show <strong>why</strong> the model made its prediction,
          not just what it decided. Grad-CAM highlights which pixels drove the result.
          The SHAP chart shows which colour channels contributed most.
        </div>
        <div class="xai-block">
          <div class="xai-block-title">
            <span class="dot" style="background:#E07B2A"></span>
            Grad-CAM &mdash; attention heatmap on ECG image
          </div>
          <img id="gradcamImg" src="" alt="Grad-CAM heatmap"/>
          <div class="xai-explain">
            <div class="xai-explain-title">What does this mean in plain English?</div>
            <p id="gcamExplain"></p>
          </div>
        </div>
        <div class="xai-block">
          <div class="xai-block-title">
            <span class="dot" style="background:#5B7FD4"></span>
            SHAP &mdash; feature attribution (integrated gradients)
          </div>
          <img id="shapImg" src="" alt="SHAP attribution"/>
          <div class="xai-explain">
            <div class="xai-explain-title">What does this mean in plain English?</div>
            <p id="shapExplain"></p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="disclaimer">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
      <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>
    <span><strong>Medical disclaimer:</strong> CardioScan is an AI research tool and is not a substitute for professional medical diagnosis. Always consult a qualified cardiologist or physician for clinical decisions.</span>
  </div>
</main>

<script>
var fileInput   = document.getElementById('fileInput');
var dropZone    = document.getElementById('dropZone');
var previewWrap = document.getElementById('preview-wrap');
var previewImg  = document.getElementById('preview-img');
var analyzeBtn  = document.getElementById('analyzeBtn');
var loading     = document.getElementById('loading');
var result      = document.getElementById('result');
var selectedFile = null;

function formatBytes(b){ return b<1048576?(b/1024).toFixed(1)+' KB':(b/1048576).toFixed(1)+' MB'; }

function showPreview(file){
  selectedFile = file;
  previewImg.src = URL.createObjectURL(file);
  document.getElementById('fname').textContent = file.name;
  document.getElementById('fsize').textContent = formatBytes(file.size);
  previewWrap.classList.add('visible');
  analyzeBtn.disabled = false;
  result.style.display = 'none';
  result.classList.remove('visible');
}

fileInput.addEventListener('change', function(e){ if(e.target.files[0]) showPreview(e.target.files[0]); });
dropZone.addEventListener('dragover',  function(e){ e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', function(){ dropZone.classList.remove('drag-over'); });
dropZone.addEventListener('drop', function(e){
  e.preventDefault(); dropZone.classList.remove('drag-over');
  var f = e.dataTransfer.files[0];
  if(f && f.type.indexOf('image/') === 0) showPreview(f);
});

function toggleXAI(){
  document.getElementById('xaiBody').classList.toggle('open');
  document.getElementById('xaiArrow').classList.toggle('open');
}

function renderResult(data){
  var pct  = data.confidence;
  var r    = 34;
  var circ = 2 * Math.PI * r;
  var dash = (pct/100) * circ;

  document.getElementById('resultHeader').innerHTML =
    '<div class="diagnosis-block">' +
      '<span class="diagnosis-pill" style="background:' + data.color + '">' + data.predicted_class + '</span>' +
      '<div>' +
        '<div class="diagnosis-label">' + data.description + '</div>' +
        '<div class="diagnosis-sub">Primary diagnosis</div>' +
      '</div>' +
    '</div>' +
    '<div class="confidence-ring">' +
      '<svg viewBox="0 0 80 80">' +
        '<circle cx="40" cy="40" r="' + r + '" fill="none" stroke="#E2E5F0" stroke-width="6"/>' +
        '<circle cx="40" cy="40" r="' + r + '" fill="none" stroke="' + data.color + '" stroke-width="6"' +
          ' stroke-dasharray="' + dash.toFixed(1) + ' ' + circ.toFixed(1) + '"' +
          ' stroke-dashoffset="' + (circ/4).toFixed(1) + '" stroke-linecap="round"/>' +
        '<text x="40" y="37" text-anchor="middle" class="val">' + pct + '%</text>' +
        '<text x="40" y="50" text-anchor="middle" class="lbl">confidence</text>' +
      '</svg>' +
    '</div>';

  var ab = document.getElementById('adviceBanner');
  ab.textContent = data.advice;
  ab.style.background  = data.color + '12';
  ab.style.borderColor = data.color;
  ab.style.color       = data.color;

  var sorted = Object.entries(data.probabilities).sort(function(a,b){ return b[1]-a[1]; });
  var barsHtml = '';
  for(var i=0; i<sorted.length; i++){
    var cls  = sorted[i][0];
    var prob = sorted[i][1];
    barsHtml +=
      '<div class="prob-row">' +
        '<span class="prob-label">' + cls + '</span>' +
        '<div class="prob-track"><div class="prob-fill" style="width:' + prob + '%;background:' + data.colors[cls] + '"></div></div>' +
        '<span class="prob-val">' + prob + '%</span>' +
      '</div>';
  }
  document.getElementById('probBars').innerHTML = barsHtml;

  document.getElementById('waveImg').src    = 'data:image/png;base64,' + data.waveform_img;
  document.getElementById('gradcamImg').src = 'data:image/png;base64,' + data.gradcam_img;
  document.getElementById('shapImg').src    = 'data:image/png;base64,' + data.shap_img;
  document.getElementById('gcamExplain').textContent  = data.gcam_explain;
  document.getElementById('shapExplain').textContent  = data.shap_explain;

  result.style.display = 'block';
  setTimeout(function(){ result.classList.add('visible'); }, 10);
  result.scrollIntoView({ behavior:'smooth', block:'start' });
}

function analyze(){
  if(!selectedFile) return;
  analyzeBtn.disabled = true;
  loading.classList.add('visible');
  result.classList.remove('visible');
  var fd = new FormData();
  fd.append('file', selectedFile);
  fetch('/predict', { method:'POST', body:fd })
    .then(function(res){ return res.json(); })
    .then(function(data){
      if(data.error){ alert('Error: ' + data.error); }
      else { renderResult(data); }
    })
    .catch(function(e){ alert('Could not connect to server. Make sure app.py is running.'); })
    .finally(function(){ loading.classList.remove('visible'); analyzeBtn.disabled = false; });
}
</script>
</body>
</html>"""

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        img    = Image.open(f.stream)
        result = predict_from_pil(img)
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  CardioScan - ECG Analysis + XAI")
    print("  Open in browser: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
