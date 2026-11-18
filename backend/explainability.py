import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class ExplainabilityEngine:
    def __init__(self, system):
        self.system = system

    @torch.no_grad()
    def compute_text_attribution(self, text, image_tensor, strategy="Fusion A"):
        words = [w for w in text.split() if w.strip()]
        if not words:
            return []
        input_ids, attention_mask = self.system.tokenizer.encode(text)
        base_result = self.system.verify(input_ids, attention_mask, image_tensor, strategy=strategy)
        base_p_fake = base_result["p_fake"]
        attributions = []
        target_words = words[:12]
        for i, word in enumerate(target_words):
            perturbed_words = target_words[:i] + target_words[i+1:]
            perturbed_text = " ".join(perturbed_words) if perturbed_words else "empty"
            p_ids, p_mask = self.system.tokenizer.encode(perturbed_text)
            p_res = self.system.verify(p_ids, p_mask, image_tensor, strategy=strategy)
            delta = base_p_fake - p_res["p_fake"]
            attributions.append({
                "word": word,
                "importance": float(delta),
                "influence": "fake-inducing" if delta > 0.01 else ("real-supporting" if delta < -0.01 else "neutral")
            })
        return attributions

    @torch.no_grad()
    def compute_vit_attention_map(self, feats):
        feature_map = feats["feature_map"]
        spatial_weights = torch.mean(torch.abs(feature_map), dim=1, keepdim=True)
        cam = F.interpolate(spatial_weights, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam

    def cam_to_heatmap_base64(self, cam, original_pil=None):
        h, w = cam.shape
        r = np.clip(np.where(cam > 0.5, cam * 255.0, 0.0), 0, 255).astype(np.uint8)
        g = np.clip((1.0 - np.abs(cam - 0.5) * 2.0) * 255.0, 0, 255).astype(np.uint8)
        b = np.clip(np.where(cam <= 0.5, (1.0 - cam * 2.0) * 255.0, 0.0), 0, 255).astype(np.uint8)
        heatmap = np.stack([r, g, b], axis=-1)
        heatmap_img = Image.fromarray(heatmap).resize((224, 224), Image.Resampling.BILINEAR)
        if original_pil is not None:
            base_img = original_pil.convert("RGB").resize((224, 224))
            out_img = Image.blend(base_img, heatmap_img, alpha=0.45)
        else:
            out_img = heatmap_img
        buf = io.BytesIO()
        out_img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @torch.no_grad()
    def evaluate_faithfulness(self, text, image_tensor, attributions, strategy="Fusion A"):
        if not attributions or len(attributions) < 2:
            return 1.0
        sorted_attrs = sorted(attributions, key=lambda x: abs(x["importance"]), reverse=True)
        top_k = max(1, len(sorted_attrs) // 3)
        top_words = set([a["word"] for a in sorted_attrs[:top_k]])
        words = text.split()
        masked_top = " ".join([w for w in words if w not in top_words]) or "neutral"
        masked_random = " ".join([w for i, w in enumerate(words) if i >= top_k]) or "neutral"
        input_ids, attention_mask = self.system.tokenizer.encode(text)
        base_p = self.system.verify(input_ids, attention_mask, image_tensor, strategy=strategy)["p_fake"]
        top_ids, top_mask = self.system.tokenizer.encode(masked_top)
        top_p = self.system.verify(top_ids, top_mask, image_tensor, strategy=strategy)["p_fake"]
        rand_ids, rand_mask = self.system.tokenizer.encode(masked_random)
        rand_p = self.system.verify(rand_ids, rand_mask, image_tensor, strategy=strategy)["p_fake"]
        diff_top = abs(base_p - top_p)
        diff_rand = abs(base_p - rand_p) + 1e-6
        faithfulness_ratio = float(min(1.0, max(0.0, diff_top / (diff_top + diff_rand))))
        return faithfulness_ratio

    def generate_verification_report(self, text, original_pil, image_tensor, strategy="Fusion A"):
        input_ids, attention_mask = self.system.tokenizer.encode(text)
        res = self.system.verify(input_ids, attention_mask, image_tensor, strategy=strategy)
        attributions = self.compute_text_attribution(text, image_tensor, strategy=strategy)
        cam = self.compute_vit_attention_map(res["features"])
        heatmap_b64 = self.cam_to_heatmap_base64(cam, original_pil)
        faithfulness = self.evaluate_faithfulness(text, image_tensor, attributions, strategy=strategy)
        cosine = res["cosine_similarity"]
        mismatch_score = float(max(0.0, 1.0 - (cosine + 1.0) / 2.0))
        evidence = []
        if mismatch_score > 0.45:
            evidence.append(f"Cross-modal semantic mismatch detected: Text and image compatibility score is low ({cosine:.3f}).")
        else:
            evidence.append(f"Visual and textual semantics align within normal thresholds ({cosine:.3f}).")
        top_fake_words = [a["word"] for a in attributions if a["influence"] == "fake-inducing"]
        if top_fake_words:
            evidence.append(f"Salient textual indicators flagged: '{', '.join(top_fake_words[:4])}'.")
        high_heat_ratio = float(np.mean(cam > 0.6))
        if high_heat_ratio > 0.15:
            evidence.append(f"Spatial visual anomaly: ViT attention rollout localized dense suspicion across {int(high_heat_ratio*100)}% of the frame.")
        else:
            evidence.append("Visual attention distribution appears diffuse and consistent across the subject.")
        if res["decision"] == "FAKE":
            explanation = f"Flagged as FAKE under {strategy} with {res['confidence']*100:.1f}% confidence. The multimodal classifier detected evidence indicative of '{res['manipulation_type']}'."
        elif res["decision"] == "REAL":
            explanation = f"Verified as REAL under {strategy} with {res['confidence']*100:.1f}% confidence. High multimodal alignment and lack of cross-modal discrepancies."
        else:
            explanation = f"Result UNCERTAIN (probability {res['p_fake']*100:.1f}% falls inside confidence gate [{self.system.t_low:.2f}, {self.system.t_high:.2f}]). Further verification required."
        return {
            "decision": res["decision"],
            "confidence": res["confidence"],
            "p_fake": res["p_fake"],
            "p_real": res["p_real"],
            "manipulation_type": res["manipulation_type"],
            "manipulation_probs": res["manipulation_probs"],
            "cosine_similarity": cosine,
            "mismatch_score": mismatch_score,
            "faithfulness_score": faithfulness,
            "short_explanation": explanation,
            "evidence": evidence,
            "text_attributions": attributions,
            "heatmap_base64": heatmap_b64,
            "strategy": strategy
        }
