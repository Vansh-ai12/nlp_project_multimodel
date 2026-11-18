import os
import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_synthetic_image(pattern="real_nature", size=(224, 224)):
    img = Image.new("RGB", size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    w, h = size
    if pattern == "real_nature":
        for y in range(h):
            r = int(120 + 80 * (y / h))
            g = int(180 + 50 * (y / h))
            b = int(220 - 70 * (y / h))
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        draw.ellipse([w*0.65, h*0.1, w*0.85, h*0.3], fill=(255, 230, 100))
        draw.polygon([(0, h), (w*0.4, h*0.5), (w*0.8, h)], fill=(34, 139, 34))
        draw.polygon([(w*0.3, h), (w*0.7, h*0.4), (w, h)], fill=(46, 110, 40))
    elif pattern == "manipulated_face":
        draw.rectangle([(0, 0), (w, h)], fill=(210, 225, 240))
        draw.ellipse([w*0.25, h*0.2, w*0.75, h*0.8], fill=(235, 195, 170))
        draw.ellipse([w*0.35, h*0.4, w*0.45, h*0.48], fill=(40, 40, 40))
        draw.ellipse([w*0.55, h*0.4, w*0.65, h*0.48], fill=(40, 40, 40))
        draw.arc([w*0.38, h*0.6, w*0.62, h*0.7], start=0, end=180, fill=(180, 50, 50), width=4)
        np_img = np.array(img)
        patch = np_img[int(h*0.35):int(h*0.55), int(w*0.3):int(w*0.7)]
        np_img[int(h*0.35):int(h*0.55), int(w*0.3):int(w*0.7)] = np.clip(patch.astype(np.int16) + 45, 0, 255).astype(np.uint8)
        img = Image.fromarray(np_img)
    elif pattern == "protest_news":
        draw.rectangle([(0, 0), (w, h)], fill=(180, 180, 190))
        for i in range(12):
            bx = (i * 20) % w
            draw.rectangle([bx, h*0.5, bx+14, h], fill=(60, 70, 85))
            draw.ellipse([bx+2, h*0.42, bx+12, h*0.5], fill=(210, 180, 150))
        draw.rectangle([w*0.1, h*0.2, w*0.6, h*0.38], fill=(240, 240, 240))
        draw.text((int(w*0.15), int(h*0.25)), "CLIMATE ACT", fill=(20, 20, 20))
    elif pattern == "tech_event":
        draw.rectangle([(0, 0), (w, h)], fill=(25, 30, 45))
        draw.rectangle([w*0.2, h*0.15, w*0.8, h*0.6], fill=(40, 60, 100), outline=(0, 200, 255), width=2)
        draw.text((int(w*0.28), int(h*0.35)), "GLOBAL SUMMIT", fill=(255, 255, 255))
        draw.rectangle([w*0.45, h*0.62, w*0.55, h*0.85], fill=(160, 160, 180))
    elif pattern == "disaster_fire":
        for y in range(h):
            r = int(220 - 100 * (y / h))
            g = int(80 + 30 * (y / h))
            b = int(30)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        draw.polygon([(w*0.2, h), (w*0.4, h*0.3), (w*0.6, h)], fill=(255, 140, 0))
        draw.polygon([(w*0.4, h), (w*0.55, h*0.2), (w*0.7, h)], fill=(255, 215, 0))
    elif pattern == "space_mars":
        draw.rectangle([(0, 0), (w, h)], fill=(10, 10, 20))
        for _ in range(30):
            sx = int(np.random.randint(0, w))
            sy = int(np.random.randint(0, int(h*0.6)))
            draw.point((sx, sy), fill=(255, 255, 255))
        draw.rectangle([(0, int(h*0.65)), (w, h)], fill=(185, 75, 45))
        draw.ellipse([w*0.3, h*0.7, w*0.5, h*0.85], fill=(140, 50, 30))
    elif pattern == "space_moon":
        draw.rectangle([(0, 0), (w, h)], fill=(5, 5, 15))
        draw.ellipse([w*0.3, h*0.2, w*0.7, h*0.6], fill=(220, 220, 220))
        draw.ellipse([w*0.4, h*0.3, w*0.48, h*0.38], fill=(180, 180, 180))
        draw.ellipse([w*0.52, h*0.42, w*0.62, h*0.52], fill=(170, 170, 170))
    elif pattern == "sports_stadium":
        draw.rectangle([(0, 0), (w, int(h*0.45))], fill=(80, 140, 200))
        draw.rectangle([(0, int(h*0.45)), (w, h)], fill=(50, 160, 70))
        draw.ellipse([w*0.3, h*0.6, w*0.7, h*0.85], outline=(255, 255, 255), width=3)
    elif pattern == "medical_lab":
        draw.rectangle([(0, 0), (w, h)], fill=(235, 245, 250))
        draw.rectangle([w*0.3, h*0.2, w*0.7, h*0.75], fill=(200, 220, 240), outline=(100, 150, 200), width=2)
        draw.line([(w*0.5, h*0.2), (w*0.5, h*0.75)], fill=(100, 150, 200), width=2)
        draw.line([(w*0.3, h*0.47), (w*0.7, h*0.47)], fill=(100, 150, 200), width=2)
    else:
        draw.rectangle([(0, 0), (w, h)], fill=(220, 220, 220))
        draw.line([(0, 0), (w, h)], fill=(150, 150, 150), width=3)
        draw.line([(0, h), (w, 0)], fill=(150, 150, 150), width=3)
    return img

HARD_CASES = [
    {
        "id": "HC-01",
        "title": "Old Forest Fire Reused For Recent California Wildfire",
        "text": "Breaking: Massive out-of-control wildfire sweeps through northern California suburbs today forcing emergency evacuations.",
        "image_pattern": "disaster_fire",
        "ground_truth": "FAKE",
        "manipulation_type": "Out of Context",
        "subreddit": "r/worldnews",
        "challenge_reason": "Single-modality image detector classifies the fire as authentic photography; multimodal cross-check discovers temporal and geolocation mismatch."
    },
    {
        "id": "HC-02",
        "title": "AI Spliced Political Figure at Climate Summit",
        "text": "President delivers keynote address at 2026 International Green Energy Summit pledging 100% emission cuts.",
        "image_pattern": "manipulated_face",
        "ground_truth": "FAKE",
        "manipulation_type": "Manipulated Image",
        "subreddit": "r/politics",
        "challenge_reason": "Text is plausible diplomatic headline; high-frequency forensic branch detects spliced facial boundary and blending boundary artifacts."
    },
    {
        "id": "HC-03",
        "title": "Authentic Climate Strike Photograph",
        "text": "Thousands of student demonstrators rally outside City Hall demanding urgent climate action legislation.",
        "image_pattern": "protest_news",
        "ground_truth": "REAL",
        "manipulation_type": "Genuine / Real",
        "subreddit": "r/pics",
        "challenge_reason": "Crowd banners and text description share strong cross-modal alignment without visual splicing or textual sensationalism."
    },
    {
        "id": "HC-04",
        "title": "Satirical Space Agency Claim",
        "text": "NASA announces newly discovered alien civilization living in high-rise subterranean lunar condos.",
        "image_pattern": "space_moon",
        "ground_truth": "FAKE",
        "manipulation_type": "Satire / Parody",
        "subreddit": "r/nottheonion",
        "challenge_reason": "Visual features show genuine moon orbiter imagery, but language exhibits distinct absurd hyperbolic satire patterns."
    },
    {
        "id": "HC-05",
        "title": "Shuffled Pair: Mars Rover Picture with Sahara Drought Caption",
        "text": "Extreme prolonged drought turns central African farmland into barren red wasteland this summer season.",
        "image_pattern": "space_mars",
        "ground_truth": "FAKE",
        "manipulation_type": "False Connection",
        "subreddit": "r/science",
        "challenge_reason": "Both text and image are independently genuine in their own domain, but cross-modal CLIP cosine similarity drops to extreme mismatch."
    },
    {
        "id": "HC-06",
        "title": "Authentic Tech Developer Conference",
        "text": "Engineers unveil next-generation quantum computing cluster during the annual Global Tech Summit.",
        "image_pattern": "tech_event",
        "ground_truth": "REAL",
        "manipulation_type": "Genuine / Real",
        "subreddit": "r/technology",
        "challenge_reason": "Stage lighting and conference signage match text prompt entities consistently."
    },
    {
        "id": "HC-07",
        "title": "Medical Vaccine Breakthrough False Statistics",
        "text": "Clinical trial confirms new compound reduces viral transmission by 99.8% across three hospital trials.",
        "image_pattern": "medical_lab",
        "ground_truth": "FAKE",
        "manipulation_type": "Misleading Text",
        "subreddit": "r/health",
        "challenge_reason": "Authentic laboratory photo used as visual trust anchor for fabricated non-peer-reviewed medical efficacy claims."
    },
    {
        "id": "HC-08",
        "title": "Sports Final Overtime Victory",
        "text": "Championship match concludes with dramatic last-minute penalty strike under the floodlights.",
        "image_pattern": "sports_stadium",
        "ground_truth": "REAL",
        "manipulation_type": "Genuine / Real",
        "subreddit": "r/sports",
        "challenge_reason": "Field markings, daylight balance, and sporting arena geometry correspond tightly."
    },
    {
        "id": "HC-09",
        "title": "National Park Autumn Foliage",
        "text": "Peaceful morning mist over the mountain ridge in Rocky Mountain National Park.",
        "image_pattern": "real_nature",
        "ground_truth": "REAL",
        "manipulation_type": "Genuine / Real",
        "subreddit": "r/earthporn",
        "challenge_reason": "Natural gradient textures with zero forensic tampering artifacts and coherent scenic caption."
    },
    {
        "id": "HC-10",
        "title": "Doctored Celebrity Scandal in Paris",
        "text": "Famous pop star caught in secret rendezvous outside Parisian hotel early Sunday morning.",
        "image_pattern": "manipulated_face",
        "ground_truth": "FAKE",
        "manipulation_type": "Manipulated Image",
        "subreddit": "r/entertainment",
        "challenge_reason": "Face swapping deepfake model generated uneven edge blurring around cheek and collarbone."
    },
    {
        "id": "HC-11",
        "title": "Film Set Explosion Claimed as Real Attack",
        "text": "Terrifying explosion rocks downtown harbor district as security forces respond to incident.",
        "image_pattern": "disaster_fire",
        "ground_truth": "FAKE",
        "manipulation_type": "Out of Context",
        "subreddit": "r/conspiracy",
        "challenge_reason": "Real cinematic pyrotechnic photo misrepresented as a terrorist disaster."
    },
    {
        "id": "HC-12",
        "title": "Satirical Onion Headline: Scientists Invent Water 2.0",
        "text": "Bottled beverage company announces revolutionary H3O formula providing double hydration per sip.",
        "image_pattern": "medical_lab",
        "ground_truth": "FAKE",
        "manipulation_type": "Satire / Parody",
        "subreddit": "r/theonion",
        "challenge_reason": "Scientific laboratory visual is authentic stock image, but textual concept violates elementary chemistry."
    },
    {
        "id": "HC-13",
        "title": "Mars Rover Rock Sample Retrieval",
        "text": "Robotic exploration rover successfully secures volcanic rock core sample on Martian crater bed.",
        "image_pattern": "space_mars",
        "ground_truth": "REAL",
        "manipulation_type": "Genuine / Real",
        "subreddit": "r/space",
        "challenge_reason": "Planetary surface visual representations align with official mission status reporting."
    },
    {
        "id": "HC-14",
        "title": "Shuffled Pair: Stadium Crowd Paired with Protest Headline",
        "text": "Mass rebellion erupts in city center as protesters clash violently with municipal police forces.",
        "image_pattern": "sports_stadium",
        "ground_truth": "FAKE",
        "manipulation_type": "False Connection",
        "subreddit": "r/news",
        "challenge_reason": "Visual features belong clearly to a sports arena with green pitch, conflicting with violent urban protest text."
    },
    {
        "id": "HC-15",
        "title": "Stock Photo Paired with Fabricated CEO Resignation",
        "text": "Global Tech CEO abruptly steps down following surprise board vote regarding accounting anomalies.",
        "image_pattern": "tech_event",
        "ground_truth": "FAKE",
        "manipulation_type": "Misleading Text",
        "subreddit": "r/wallstreetbets",
        "challenge_reason": "Generic summit visual is paired with market-moving fraudulent corporate governance claim."
    },
    {
        "id": "HC-16",
        "title": "Subtle Face Boundary Morph in Portrait",
        "text": "Official portrait of recently appointed trade representative released today.",
        "image_pattern": "manipulated_face",
        "ground_truth": "FAKE",
        "manipulation_type": "Manipulated Image",
        "subreddit": "r/politics",
        "challenge_reason": "Mild morphing artifacts around jawline detectable primarily by forensic Laplacian edge filtering."
    }
]

def get_hard_cases_catalog(save_dir="static/samples"):
    os.makedirs(save_dir, exist_ok=True)
    catalog = []
    for item in HARD_CASES:
        img_name = f"{item['id'].lower()}_{item['image_pattern']}.jpg"
        img_path = os.path.join(save_dir, img_name)
        if not os.path.exists(img_path):
            img = create_synthetic_image(item["image_pattern"])
            img.save(img_path, format="JPEG", quality=95)
        rel_url = f"/static/samples/{img_name}"
        catalog.append({
            **item,
            "image_url": rel_url,
            "image_filename": img_name
        })
    return catalog

def get_benchmark_dataset(num_samples=60):
    dataset = []
    patterns = ["real_nature", "manipulated_face", "protest_news", "tech_event", "disaster_fire", "space_mars", "space_moon", "sports_stadium", "medical_lab"]
    for i in range(num_samples):
        case = HARD_CASES[i % len(HARD_CASES)]
        pat = case["image_pattern"]
        img = create_synthetic_image(pat)
        dataset.append({
            "id": f"BENCH-{i:03d}",
            "text": case["text"],
            "image": img,
            "pattern": pat,
            "label": 1 if case["ground_truth"] == "FAKE" else 0,
            "label_str": case["ground_truth"],
            "manipulation_type": case["manipulation_type"],
            "subreddit": case["subreddit"]
        })
    return dataset
