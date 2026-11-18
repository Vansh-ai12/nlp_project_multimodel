import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTokenizer:
    def __init__(self, vocab_size=30522):
        self.vocab_size = vocab_size

    def encode(self, text, max_len=64):
        tokens = [ord(c) % (self.vocab_size - 100) + 100 for c in text.lower().strip()[:max_len]]
        pad_len = max(0, max_len - len(tokens))
        tokens = tokens + [0] * pad_len
        tokens = tokens[:max_len]
        mask = [1 if t > 0 else 0 for t in tokens]
        return torch.tensor([tokens], dtype=torch.long), torch.tensor([mask], dtype=torch.float32)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=30522, d_model=768, clip_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(128, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.clip_projection = nn.Linear(d_model, clip_dim)

    def forward(self, input_ids, attention_mask=None):
        b, s = input_ids.shape
        pos = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        hidden = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        pooled = hidden[:, 0, :]
        clip_text_emb = F.normalize(self.clip_projection(pooled), p=2, dim=-1)
        return pooled, clip_text_emb, hidden

class ImageEncoder(nn.Module):
    def __init__(self, clip_dim=512):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_block(64, 128, stride=2)
        self.layer2 = self._make_block(128, 256, stride=2)
        self.layer3 = self._make_block(256, 512, stride=2)
        self.vit_proj = nn.Linear(512 * 7 * 7, clip_dim)

    def _make_block(self, in_c, out_c, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        h = self.relu(self.bn1(self.conv_in(x)))
        h = self.maxpool(h)
        h = self.layer1(h)
        h = self.layer2(h)
        feature_map = self.layer3(h)
        b = feature_map.shape[0]
        flat = feature_map.reshape(b, -1)
        img_emb = F.normalize(self.vit_proj(flat), p=2, dim=-1)
        return img_emb, feature_map

class ForensicBranch(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        laplacian_kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]], device=x.device).repeat(3, 1, 1, 1)
        edges = F.conv2d(x, laplacian_kernel, padding=1, groups=3)
        return self.net(edges)

class TextOnlyBaseline(nn.Module):
    def __init__(self, text_dim=768, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, text_feat):
        return self.classifier(text_feat)

class ImageOnlyBaseline(nn.Module):
    def __init__(self, img_dim=512, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, img_feat):
        return self.classifier(img_feat)

class MultimodalFusionA(nn.Module):
    def __init__(self, text_dim=768, img_dim=512, forensic_dim=64, num_classes=2, num_manip_types=6):
        super().__init__()
        in_dim = text_dim + img_dim + 1 + forensic_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.binary_head = nn.Linear(128, num_classes)
        self.manip_head = nn.Linear(128, num_manip_types)

    def forward(self, text_feat, img_feat, cosine_sim, forensic_feat):
        fused = torch.cat([text_feat, img_feat, cosine_sim, forensic_feat], dim=-1)
        hidden = self.fusion_mlp(fused)
        binary_logits = self.binary_head(hidden)
        manip_logits = self.manip_head(hidden)
        return binary_logits, manip_logits

class CrossAttention(nn.Module):
    def __init__(self, d_model=256, nhead=4):
        super().__init__()
        self.mha_t2i = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.mha_i2t = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_i = nn.LayerNorm(d_model)

    def forward(self, text_tokens, img_tokens):
        t_attended, _ = self.mha_t2i(text_tokens, img_tokens, img_tokens)
        t_out = self.norm_t(text_tokens + t_attended)
        i_attended, _ = self.mha_i2t(img_tokens, text_tokens, text_tokens)
        i_out = self.norm_i(img_tokens + i_attended)
        return t_out, i_out

class MultimodalFusionB(nn.Module):
    def __init__(self, text_dim=768, img_dim=512, forensic_dim=64, proj_dim=256, num_classes=2, num_manip_types=6):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.img_proj = nn.Linear(img_dim, proj_dim)
        self.cross_attn = CrossAttention(d_model=proj_dim, nhead=4)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim * 2 + 1 + forensic_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.binary_head = nn.Linear(128, num_classes)
        self.manip_head = nn.Linear(128, num_manip_types)

    def forward(self, text_feat, img_feat, cosine_sim, forensic_feat, modality_dropout=0.0):
        t = self.text_proj(text_feat).unsqueeze(1)
        i = self.img_proj(img_feat).unsqueeze(1)
        t_att, i_att = self.cross_attn(t, i)
        t_pooled = t_att.squeeze(1)
        i_pooled = i_att.squeeze(1)
        if self.training and modality_dropout > 0.0:
            mask_t = (torch.rand(text_feat.size(0), 1, device=text_feat.device) > modality_dropout).float()
            mask_i = (torch.rand(img_feat.size(0), 1, device=img_feat.device) > modality_dropout).float()
            t_pooled = t_pooled * mask_t
            i_pooled = i_pooled * mask_i
        fused = torch.cat([t_pooled, i_pooled, cosine_sim, forensic_feat], dim=-1)
        hidden = self.fusion_mlp(fused)
        binary_logits = self.binary_head(hidden)
        manip_logits = self.manip_head(hidden)
        return binary_logits, manip_logits

class ContentVerificationSystem(nn.Module):
    def __init__(self, temperature=1.2, t_low=0.35, t_high=0.65):
        super().__init__()
        self.tokenizer = SimpleTokenizer()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.forensic_branch = ForensicBranch()
        self.text_baseline = TextOnlyBaseline()
        self.image_baseline = ImageOnlyBaseline()
        self.fusion_a = MultimodalFusionA()
        self.fusion_b = MultimodalFusionB()
        self.temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=False)
        self.t_low = t_low
        self.t_high = t_high
        self.manip_labels = [
            "Genuine / Real",
            "Manipulated Image",
            "Misleading Text",
            "Out of Context",
            "Satire / Parody",
            "False Connection"
        ]

    def set_calibration(self, temperature=None, t_low=None, t_high=None):
        if temperature is not None:
            self.temperature.data = torch.tensor([temperature], dtype=torch.float32)
        if t_low is not None:
            self.t_low = t_low
        if t_high is not None:
            self.t_high = t_high

    def extract_features(self, input_ids, attention_mask, image_tensor):
        text_feat, clip_text_emb, hidden_text = self.text_encoder(input_ids, attention_mask)
        img_feat, feature_map = self.image_encoder(image_tensor)
        cosine_sim = F.cosine_similarity(clip_text_emb, img_feat, dim=-1).unsqueeze(-1)
        forensic_feat = self.forensic_branch(image_tensor)
        return {
            "text_feat": text_feat,
            "clip_text_emb": clip_text_emb,
            "hidden_text": hidden_text,
            "img_feat": img_feat,
            "feature_map": feature_map,
            "cosine_sim": cosine_sim,
            "forensic_feat": forensic_feat
        }

    def verify(self, input_ids, attention_mask, image_tensor, strategy="Fusion A"):
        feats = self.extract_features(input_ids, attention_mask, image_tensor)
        if strategy == "Text Only":
            logits = self.text_baseline(feats["text_feat"])
            manip_logits = torch.zeros((logits.size(0), len(self.manip_labels)), device=logits.device)
        elif strategy == "Image Only":
            logits = self.image_baseline(feats["img_feat"])
            manip_logits = torch.zeros((logits.size(0), len(self.manip_labels)), device=logits.device)
        elif strategy == "Fusion B":
            logits, manip_logits = self.fusion_b(feats["text_feat"], feats["img_feat"], feats["cosine_sim"], feats["forensic_feat"])
        else:
            logits, manip_logits = self.fusion_a(feats["text_feat"], feats["img_feat"], feats["cosine_sim"], feats["forensic_feat"])
        scaled_logits = logits / torch.clamp(self.temperature, min=0.1)
        probs = F.softmax(scaled_logits, dim=-1)
        p_fake = probs[:, 1].item()
        p_real = probs[:, 0].item()
        if p_fake < self.t_low:
            decision = "REAL"
            confidence = p_real
        elif p_fake > self.t_high:
            decision = "FAKE"
            confidence = p_fake
        else:
            decision = "UNCERTAIN"
            confidence = 1.0 - 2.0 * abs(p_fake - 0.5)
        manip_probs = F.softmax(manip_logits, dim=-1)
        manip_idx = torch.argmax(manip_probs, dim=-1).item()
        manip_type = self.manip_labels[manip_idx]
        return {
            "decision": decision,
            "confidence": float(confidence),
            "p_fake": float(p_fake),
            "p_real": float(p_real),
            "cosine_similarity": float(feats["cosine_sim"].item()),
            "manipulation_type": manip_type,
            "manipulation_probs": {self.manip_labels[i]: float(manip_probs[0, i].item()) for i in range(len(self.manip_labels))},
            "features": feats
        }
