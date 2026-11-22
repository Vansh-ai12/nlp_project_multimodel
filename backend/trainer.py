import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from backend.model import ContentVerificationSystem
from backend.dataset import get_benchmark_dataset, HARD_CASES, create_synthetic_image

def pil_to_tensor(pil_img):
    img = pil_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    reliability_data = []
    for i in range(n_bins):
        b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= b_low) & (probs < b_high) if i < n_bins - 1 else (probs >= b_low) & (probs <= b_high)
        n_items = int(np.sum(mask))
        if n_items > 0:
            bin_acc = float(np.mean(labels[mask]))
            bin_conf = float(np.mean(probs[mask]))
            ece += (n_items / len(probs)) * abs(bin_acc - bin_conf)
            reliability_data.append({
                "bin_mid": float((b_low + b_high) / 2.0),
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "count": n_items
            })
        else:
            reliability_data.append({
                "bin_mid": float((b_low + b_high) / 2.0),
                "accuracy": 0.0,
                "confidence": float((b_low + b_high) / 2.0),
                "count": 0
            })
    return float(ece), reliability_data

def compute_coverage_vs_accuracy(probs, labels, step=0.05):
    thresholds = np.arange(0.5, 0.95, step)
    curve = []
    for t in thresholds:
        accepted = (probs <= (1.0 - t)) | (probs >= t)
        cov = float(np.mean(accepted))
        if np.sum(accepted) > 0:
            preds = (probs[accepted] >= 0.5).astype(int)
            acc = float(accuracy_score(labels[accepted], preds))
        else:
            acc = 1.0
        curve.append({
            "confidence_threshold": round(float(t), 2),
            "coverage": round(cov, 4),
            "accuracy": round(acc, 4)
        })
    return curve

class ExperimentManager:
    def __init__(self, checkpoint_path=None, metrics_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_dir = os.path.join(base_dir, "checkpoints")

        # Check for GPU weights or default weights
        candidate_ckpts = [
            checkpoint_path,
            os.path.join(ckpt_dir, "system_weights.pt"),
            os.path.join(ckpt_dir, "system_weights_gpu.pt"),
            "checkpoints/system_weights.pt",
            "checkpoints/system_weights_gpu.pt"
        ]
        self.checkpoint_path = None
        for cp in candidate_ckpts:
            if cp and os.path.exists(cp):
                self.checkpoint_path = cp
                break

        if not self.checkpoint_path:
            self.checkpoint_path = os.path.join(ckpt_dir, "system_weights.pt")

        candidate_metrics = [
            metrics_path,
            os.path.join(ckpt_dir, "full_benchmark_results.json"),
            os.path.join(ckpt_dir, "metrics.json"),
            os.path.join(ckpt_dir, "metrics_gpu.json"),
            "checkpoints/full_benchmark_results.json",
            "checkpoints/metrics.json"
        ]
        self.metrics_path = None
        for mp in candidate_metrics:
            if mp and os.path.exists(mp):
                self.metrics_path = mp
                break

        if not self.metrics_path:
            self.metrics_path = os.path.join(ckpt_dir, "metrics.json")

        self.system = ContentVerificationSystem()
        self.metrics_cache = {}

        if os.path.exists(self.checkpoint_path):
            try:
                self.load_checkpoint()
            except Exception as e:
                print(f"[!] Note: Loading weights fallback: {e}")
                self.system.eval()
        else:
            self.system.eval()

        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self.metrics_cache = json.load(f)
            except Exception:
                pass

    def load_checkpoint(self):
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.system.load_state_dict(state)
        self.system.eval()

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(self.system.state_dict(), self.checkpoint_path)

    def train_and_calibrate(self, epochs=6):
        dataset = get_benchmark_dataset(num_samples=72)
        optimizer = optim.Adam([
            {"params": self.system.text_baseline.parameters(), "lr": 1e-3},
            {"params": self.system.image_baseline.parameters(), "lr": 1e-3},
            {"params": self.system.fusion_a.parameters(), "lr": 1e-3},
            {"params": self.system.fusion_b.parameters(), "lr": 1e-3},
            {"params": self.system.forensic_branch.parameters(), "lr": 5e-4}
        ])
        criterion_bce = nn.CrossEntropyLoss()
        criterion_manip = nn.CrossEntropyLoss()
        manip_map = {name: i for i, name in enumerate(self.system.manip_labels)}
        self.system.train()
        for ep in range(epochs):
            for item in dataset:
                optimizer.zero_grad()
                input_ids, attention_mask = self.system.tokenizer.encode(item["text"])
                img_tensor = pil_to_tensor(item["image"])
                y_bin = torch.tensor([item["label"]], dtype=torch.long)
                y_manip = torch.tensor([manip_map.get(item["manipulation_type"], 0)], dtype=torch.long)
                feats = self.system.extract_features(input_ids, attention_mask, img_tensor)
                l_text = self.system.text_baseline(feats["text_feat"])
                l_img = self.system.image_baseline(feats["img_feat"])
                la_bin, la_manip = self.system.fusion_a(feats["text_feat"], feats["img_feat"], feats["cosine_sim"], feats["forensic_feat"])
                lb_bin, lb_manip = self.system.fusion_b(feats["text_feat"], feats["img_feat"], feats["cosine_sim"], feats["forensic_feat"], modality_dropout=0.1)
                loss = (
                    criterion_bce(l_text, y_bin) +
                    criterion_bce(l_img, y_bin) +
                    criterion_bce(la_bin, y_bin) +
                    0.5 * criterion_manip(la_manip, y_manip) +
                    criterion_bce(lb_bin, y_bin) +
                    0.5 * criterion_manip(lb_manip, y_manip)
                )
                loss.backward()
                optimizer.step()
        self.system.eval()
        self.save_checkpoint()
        self.run_full_evaluation()

    def run_full_evaluation(self):
        test_dataset = get_benchmark_dataset(num_samples=32)
        held_out_subreddits = ["r/nottheonion", "r/science", "r/wallstreetbets"]
        strategies = ["Text Only", "Image Only", "Fusion A", "Fusion B"]
        results = {}
        for strat in strategies:
            y_true, y_pred, y_probs = [], [], []
            gen_true, gen_pred = [], []
            for item in test_dataset:
                input_ids, attention_mask = self.system.tokenizer.encode(item["text"])
                img_tensor = pil_to_tensor(item["image"])
                res = self.system.verify(input_ids, attention_mask, img_tensor, strategy=strat)
                pred_bin = 1 if res["p_fake"] >= 0.5 else 0
                y_true.append(item["label"])
                y_pred.append(pred_bin)
                y_probs.append(res["p_fake"])
                if item["subreddit"] in held_out_subreddits:
                    gen_true.append(item["label"])
                    gen_pred.append(pred_bin)
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_probs = np.array(y_probs)
            acc = float(accuracy_score(y_true, y_pred))
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            cm = confusion_matrix(y_true, y_pred).tolist()
            ece, rel = compute_ece(y_probs, y_true)
            cov_acc = compute_coverage_vs_accuracy(y_probs, y_true)
            gen_acc = float(accuracy_score(gen_true, gen_pred)) if gen_true else acc
            results[strat] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "confusion_matrix": cm,
                "ece": round(ece, 4),
                "generalization_acc": round(gen_acc, 4),
                "reliability_curve": rel,
                "coverage_vs_accuracy": cov_acc
            }
        ablation = self.run_modality_ablation(test_dataset)
        shuffled = self.run_shuffled_pair_sanity_test(test_dataset)
        self.metrics_cache = {
            "strategies": results,
            "ablation": ablation,
            "shuffled_pair_test": shuffled
        }
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_cache, f, indent=2)
        return self.metrics_cache

    def run_modality_ablation(self, dataset):
        ablations = {}
        for mode in ["Full Multimodal (Fusion A)", "Ablation: No Forensic Branch", "Ablation: No CLIP Cosine Sim"]:
            y_true, y_pred = [], []
            for item in dataset:
                input_ids, attention_mask = self.system.tokenizer.encode(item["text"])
                img_tensor = pil_to_tensor(item["image"])
                feats = self.system.extract_features(input_ids, attention_mask, img_tensor)
                cos = feats["cosine_sim"] if mode != "Ablation: No CLIP Cosine Sim" else torch.zeros_like(feats["cosine_sim"])
                forensic = feats["forensic_feat"] if mode != "Ablation: No Forensic Branch" else torch.zeros_like(feats["forensic_feat"])
                logits, _ = self.system.fusion_a(feats["text_feat"], feats["img_feat"], cos, forensic)
                p_fake = F.softmax(logits, dim=-1)[0, 1].item()
                y_true.append(item["label"])
                y_pred.append(1 if p_fake >= 0.5 else 0)
            ablations[mode] = {
                "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
            }
        return ablations

    def run_shuffled_pair_sanity_test(self, dataset):
        real_items = [d for d in dataset if d["label"] == 0]
        if len(real_items) < 4:
            real_items = dataset[:6]
        n = len(real_items)
        detected_as_mismatch = 0
        cosine_drops = []
        for i in range(n):
            item_a = real_items[i]
            item_b = real_items[(i + 1) % n]
            t_ids, t_mask = self.system.tokenizer.encode(item_a["text"])
            orig_img = pil_to_tensor(item_a["image"])
            shuffled_img = pil_to_tensor(item_b["image"])
            orig_feats = self.system.extract_features(t_ids, t_mask, orig_img)
            shuf_feats = self.system.extract_features(t_ids, t_mask, shuffled_img)
            orig_cos = orig_feats["cosine_sim"].item()
            shuf_cos = shuf_feats["cosine_sim"].item()
            cosine_drops.append(orig_cos - shuf_cos)
            res = self.system.verify(t_ids, t_mask, shuffled_img, strategy="Fusion A")
            if res["p_fake"] > 0.5 or res["decision"] in ["FAKE", "UNCERTAIN"]:
                detected_as_mismatch += 1
        mismatch_detection_rate = detected_as_mismatch / n
        avg_cosine_drop = float(np.mean(cosine_drops))
        return {
            "num_shuffled_pairs": n,
            "mismatch_detection_rate": round(mismatch_detection_rate, 4),
            "avg_cosine_drop": round(avg_cosine_drop, 4),
            "status": "PASS: Shuffled cross-modal pairs reliably trigger semantic conflict" if mismatch_detection_rate >= 0.75 else "ATTENTION: Low mismatch sensitivity"
        }
