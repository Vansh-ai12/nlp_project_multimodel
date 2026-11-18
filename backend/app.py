import os
import io
import base64
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from backend.trainer import ExperimentManager, pil_to_tensor
from backend.explainability import ExplainabilityEngine
from backend.dataset import get_hard_cases_catalog, HARD_CASES, create_synthetic_image

app = FastAPI(title="Multimodal Deepfake Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/samples", exist_ok=True)
get_hard_cases_catalog("static/samples")

manager = ExperimentManager()
explainer = ExplainabilityEngine(manager.system)

app.mount("/static", StaticFiles(directory="static"), name="static")

class VerificationPayload(BaseModel):
    text: str
    strategy: Optional[str] = "Fusion A"
    temperature: Optional[float] = 1.2
    t_low: Optional[float] = 0.35
    t_high: Optional[float] = 0.65
    sample_id: Optional[str] = None
    image_base64: Optional[str] = None

class ComparePayload(BaseModel):
    text: str
    sample_id: Optional[str] = None
    image_base64: Optional[str] = None

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "device": "cpu",
        "models_loaded": ["Text Only Baseline", "Image Only Baseline", "Multimodal Fusion A", "Multimodal Fusion B"],
        "checkpoint_exists": os.path.exists(manager.checkpoint_path),
        "calibration": {
            "temperature": float(manager.system.temperature.item()),
            "t_low": manager.system.t_low,
            "t_high": manager.system.t_high
        }
    }

@app.get("/api/hard-cases")
def list_hard_cases():
    catalog = get_hard_cases_catalog("static/samples")
    return {"hard_cases": catalog}

def resolve_image(sample_id: Optional[str], image_base64: Optional[str]) -> Image.Image:
    if image_base64:
        try:
            data = image_base64
            if "," in data:
                data = data.split(",", 1)[1]
            raw = base64.b64decode(data)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            pass
    if sample_id:
        catalog = {item["id"]: item for item in HARD_CASES}
        if sample_id in catalog:
            return create_synthetic_image(catalog[sample_id]["image_pattern"])
    return create_synthetic_image("real_nature")

@app.post("/api/verify")
def verify_content(payload: VerificationPayload):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text caption cannot be empty.")
    pil_img = resolve_image(payload.sample_id, payload.image_base64)
    manager.system.set_calibration(
        temperature=payload.temperature,
        t_low=payload.t_low,
        t_high=payload.t_high
    )
    img_tensor = pil_to_tensor(pil_img)
    strategy = payload.strategy or "Fusion A"
    report = explainer.generate_verification_report(text, pil_img, img_tensor, strategy=strategy)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    report["input_image_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
    return report

@app.post("/api/compare-all")
def compare_all_baselines(payload: ComparePayload):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text caption cannot be empty.")
    pil_img = resolve_image(payload.sample_id, payload.image_base64)
    img_tensor = pil_to_tensor(pil_img)
    input_ids, attention_mask = manager.system.tokenizer.encode(text)
    strats = ["Text Only", "Image Only", "Fusion A", "Fusion B"]
    comparison = {}
    last_cos = 0.0
    for s in strats:
        res = manager.system.verify(input_ids, attention_mask, img_tensor, strategy=s)
        last_cos = res["cosine_similarity"]
        comparison[s] = {
            "decision": res["decision"],
            "confidence": res["confidence"],
            "p_fake": res["p_fake"],
            "p_real": res["p_real"],
            "manipulation_type": res["manipulation_type"] if s.startswith("Fusion") else "N/A (Single Modality)"
        }
    return {"comparison": comparison, "cosine_similarity": float(last_cos)}

@app.get("/api/metrics")
def get_metrics():
    if not manager.metrics_cache:
        manager.run_full_evaluation()
    return manager.metrics_cache

@app.post("/api/retrain")
def retrain_models(epochs: int = 6):
    manager.train_and_calibrate(epochs=epochs)
    return {"status": "retraining completed", "metrics": manager.metrics_cache}
