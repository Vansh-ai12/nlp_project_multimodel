# Multimodal Deepfake Detection: Explainable Content Verification

## Authors
MRIDUL BISHT (S24CSEU0208) · MANU SHARMA (S24CSEU0106) · VANSH JAIN (S24CSU0161) · MEET ANAND (S24CSEU0277) · UTKARSH SINGH (S24CSEU0096) · ROHAN GARG (S24CSE0371)

## Architecture Overview
- **Text Modality Pipeline**: DistilBERT 768-d semantic representation + CLIP Text 512-d cross-modal embedding
- **Visual Modality Pipeline**: CLIP ViT-B/32 512-d visual concept embedding + 64-d Laplacian forensic residual branch
- **Multimodal Fusion A**: Feature concatenation with explicit CLIP text-image cosine similarity alignment
- **Multimodal Fusion B**: Bidirectional cross-attention (Text-to-Image and Image-to-Text) with modality dropout
- **Calibrated Decision Gate**: Temperature-scaled softmax with three output states:
  - `REAL`: p(fake) < t_low
  - `UNCERTAIN`: t_low <= p(fake) <= t_high
  - `FAKE`: p(fake) > t_high
- **Auxiliary Manipulation-Type Head**: 6-way classification (Genuine/Real, Manipulated Image, Misleading Text, Out of Context, Satire/Parody, False Connection)
- **Explainability Engine**:
  - Text: LIME token perturbation attribution
  - Visual: ViT attention rollout / Grad-CAM spatial activation heatmap overlay
  - Cross-modal mismatch metric
  - Faithfulness evaluation ratio

## Project Structure
- `backend/model.py`: Neural architectures (Encoders, Baselines, Fusion A, Fusion B, Calibration)
- `backend/explainability.py`: Explainability engine (LIME tokens, ViT heatmaps, Faithfulness)
- `backend/dataset.py`: Benchmark dataset generator and 16 logged hard cases
- `backend/trainer.py`: Training loop, evaluation, ECE calibration, and baseline comparisons
- `backend/app.py`: FastAPI application serving REST endpoints and web dashboard
- `static/index.html`: Web dashboard UI
- `static/style.css`: Modern responsive UI stylesheet
- `static/app.js`: Dynamic frontend controller
- `run.py`: Server launcher script

## How to Run the Project
1. Install dependencies:
```bash
pip install torch transformers fastapi uvicorn pillow scikit-learn numpy
```

2. Start the application:
```bash
python run.py
```

3. Open your browser:
Navigate to `http://localhost:8000`
