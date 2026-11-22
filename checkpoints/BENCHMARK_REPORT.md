# Multimodal NLP Deepfake Detection: Comprehensive Course Benchmark Report

**Course:** CSET 346 - Natural Language Processing  
**Compute Hardware:** NVIDIA DGX H200 (140 GB HBM3e VRAM)  

## 1. Experiment 1: Single-Modality Baselines vs Multimodal Architecture

| Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Text-Only Baseline** | 49.00% | 49.09% | 54.00% | 51.43% | 49.20% |
| **Image-Only Baseline** | 50.00% | 50.00% | 56.00% | 52.83% | 56.72% |
| **Multimodal (Ours)** | 70.00% | 72.73% | 64.00% | 68.09% | 77.92% |

## 2. Experiment 2: Fusion Strategy Comparison

| Strategy | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Strategy A: Feature Concatenation** | 50.00% | 0.00% | 79.48% |
| **Strategy B: Differential Cross-Attention (DCA)** | 70.00% | 68.09% | 77.92% |

## 3. Experiment 3: Generalization (Seen vs Unseen Manipulations)

| Evaluation Split | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **In-Distribution (Seen Manipulation Types)** | 73.33% | 61.54% | 75.60% |
| **Out-of-Distribution (Unseen Manipulations FA & TA)** | 64.00% | 58.46% | 73.68% |

## 4. Experiment 4: Modality Ablation Study

| Ablation Variant | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Full Multimodal Model** | 70.00% | 68.09% | 77.92% |
| **Ablation: Without Forensic Branch (PRNU/FFT)** | 64.00% | 70.97% | 73.64% |
| **Ablation: Without Cross-Modal Cosine Alignment** | 50.00% | 0.00% | 64.00% |
| **Ablation: Without Both Forensics & Cosine** | 50.00% | 0.00% | 52.32% |
| **Ablation: Text Only (Zero Vision)** | 49.00% | 51.43% | 49.20% |
| **Ablation: Image Only (Zero Text)** | 50.00% | 52.83% | 56.72% |

## 5. Experiment 5: Qualitative Error Analysis (16 Hard Cases)

**Overall Edge-Case Accuracy:** 13/16 (81.2%)

| ID | Case Title | Domain | Manipulation Type | Ground Truth | Prediction | Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HC-01` | Wildfire Recontextualization | Disaster News | Text Attribute (TA) | FAKE | **FAKE** | 95.9% |
| `HC-02` | Diplomatic Summit Deepfake | Politics | Face Swap (FS) | FAKE | **FAKE** | 94.1% |
| `HC-03` | Climate Rally Coverage | Social/Environment | Pristine / Genuine | REAL | **FAKE** | 92.1% |
| `HC-04` | Lunar Space Discovery Satire | Science/Satire | Text Swap (TS) | FAKE | **REAL** | 66.4% |
| `HC-05` | Mars Rover vs African Drought | Environment | Text Swap (TS) | FAKE | **FAKE** | 69.5% |
| `HC-06` | Tech Keynote Address | Technology | Pristine / Genuine | REAL | **REAL** | 83.5% |
| `HC-07` | Clinical Efficacy Exaggeration | Medical | Text Attribute (TA) | FAKE | **FAKE** | 98.6% |
| `HC-08` | Championship Match Victory | Sports | Pristine / Genuine | REAL | **REAL** | 69.2% |
| `HC-09` | National Park Autumn Mist | Nature | Pristine / Genuine | REAL | **FAKE** | 74.1% |
| `HC-10` | Celebrity Hotel Splicing | Entertainment | Face Swap (FS) | FAKE | **FAKE** | 97.8% |
| `HC-11` | Movie Pyrotechnics as Attack | Conflict | Text Attribute (TA) | FAKE | **FAKE** | 78.1% |
| `HC-12` | Satirical Water 2.0 Formula | Satire | Text Swap (TS) | FAKE | **FAKE** | 74.0% |
| `HC-13` | Martian Crater Core Sample | Space | Pristine / Genuine | REAL | **REAL** | 89.5% |
| `HC-14` | Stadium Crowd as Street Riot | News | Text Swap (TS) | FAKE | **FAKE** | 66.9% |
| `HC-15` | Corporate CEO Resignation | Finance | Text Attribute (TA) | FAKE | **FAKE** | 90.2% |
| `HC-16` | Portrait Action Unit Morph | Politics | Face Attribute (FA) | FAKE | **FAKE** | 86.4% |

## 6. Experiment 6: Explanation Faithfulness Evaluation

- **Top-Attributed Feature Confidence Drop:** `0.1800`
- **Random Control Feature Confidence Drop:** `0.0589`
- **Faithfulness Ratio:** `0.7536`
- **Verdict:** High attribution faithfulness: masking top evidence tokens degrades model prediction 2.3x more than random masking.

## 7. Cross-Modal Sanity Check

- **Shuffled-Pair Detection Rate:** `80.0%`
- **Result:** `PASS: Model reliably detects cross-modal semantic contradictions.`
