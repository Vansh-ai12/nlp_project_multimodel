# Multimodal Deepfake Testing & Data Guide

This guide explains where to get data to test the system across all four manipulation categories: **Genuine/Real**, **Visual Manipulation (Deepfakes)**, **Out-of-Context / False Connection (Text-Image Mismatch)**, and **Machine-Generated Text**.

---

## 1. Instant 1-Click Testing (Built Into Your Web Dashboard)

Your web dashboard at `http://localhost:8000` has **16 pre-loaded grounded test cases** already generated in `static/samples/`.

At the top of the dashboard, use the dropdown:
**"⚡ Load Curated Hard-Case Scenarios (Experiment 5 Benchmark)"**

| Case ID | Scenario Title | Domain | Manipulation Type | Ground Truth | What It Tests |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `HC-01` | Wildfire Recontextualization | Disaster News | Text Attribute (TA) | **FAKE** | Real fire photo with exaggerated casualty caption |
| `HC-02` | Diplomatic Summit Deepfake | Politics | Face Swap (FS) | **FAKE** | Spliced facial boundary artifacts (Laplacian branch) |
| `HC-03` | Climate Strike Rally | Social | **REAL** | **REAL** | Authentic protest photo + coherent caption |
| `HC-04` | Lunar Discovery Claim | Science / Satire | Text Swap (TS) | **FAKE** | Authentic moon image + absurd alien discovery text |
| `HC-05` | Mars Rover vs African Drought | Environment | False Connection | **FAKE** | Mars red soil paired with Sahara drought claim |
| `HC-06` | Global Tech Summit | Technology | **REAL** | **REAL** | Genuine conference signage + consistent text |
| `HC-07` | Medical Trial Statistics | Health / Medical | Text Attribute (TA) | **FAKE** | Authentic lab photo with fabricated statistical efficacy |
| `HC-08` | Stadium Match Victory | Sports | **REAL** | **REAL** | Authentic stadium photography |
| `HC-10` | Celebrity Paparazzi Splicing | Entertainment | Face Swap (FS) | **FAKE** | Spliced face with edge blurring artifacts |
| `HC-14` | Stadium as Urban Riot | News | False Connection | **FAKE** | Sports pitch paired with violent riot caption |

Clicking **"Load Selected Case"** autofills both the image and the text caption immediately.

---

## 2. Real-World Public Benchmark Datasets (For Submission to Dr. Sakshi)

If you want to test or showcase large-scale real datasets as recommended in `PJ_07.pdf`:

### A. Fakeddit (Primary Course Dataset Recommendation)
* **What it is:** The largest multimodal misinformation dataset (over 1 million Reddit image-text pairs with 2-way, 3-way, and 6-way labels).
* **Where to get it:**
  * GitHub: `https://github.com/entitize/Fakeddit`
  * HuggingFace: `https://huggingface.co/datasets/rungalileo/fakeddit`
* **How to test a sample:** Download any image and its Reddit title from the dataset, paste the title into the dashboard, upload the image, and click **"Analyze & Verify"**.

### B. DGM4 (From CVPR 2023 Paper by Shao et al.)
* **What it is:** 230,000 human-centric news image-text pairs from The Guardian, BBC, and Washington Post with Face Swap (SimSwap/InfoSwap), Face Attribute (StyleCLIP), Text Swap, and Text Attribute manipulations.
* **Where to get it:**
  * GitHub: `https://github.com/rshaojimmy/MultiModal-DeepFake`

### C. Which Face Is Real? (StyleGAN2 Human vs AI Faces)
* **What it is:** High-resolution paired human portraits vs. StyleGAN2 synthesized faces created by Jevin West and Carl Bergstrom (University of Washington).
* **Where to get it:** `https://www.whichfaceisreal.com/`
* **How to test:** Right-click and save an AI-generated face and a real human face. Upload to the studio:
  * The real photo will show authentic camera sensor noise.
  * The AI photo will trigger the **2D FFT spectral power peaks** and show zero camera PRNU.

### D. TweepFake (Twitter Deepfake Tweets)
* **What it is:** 25,572 tweets from human accounts vs 23 AI bots (GPT-2, Markov, RNN).
* **Where to get it:**
  * Kaggle: `https://www.kaggle.com/datasets/mtesconi/twitter-deep-fake-text`

---

## 3. How to Test "In-The-Wild" Cases Manually (Interactive Testing)

You can test any image and caption from the web using these three test archetypes:

### Test Case 1: The "Out-of-Context" Test (Semantic Mismatch)
1. Find any real photo of a major sports event (e.g., FIFA World Cup stadium or cricket match).
2. Enter this caption:
   `"Massive civil unrest and violent clashes erupt in downtown Paris as emergency forces deploy tear gas today."`
3. Click **"Analyze & Verify"**:
   * The **Differential Cross-Modal Attention (DCA)** module will flag the extreme semantic distance between sports field visual features and civil unrest text, classifying it as **`FAKE` (False Connection / Out of Context)** with high confidence.

### Test Case 2: The "Pristine News" Test (Authentic)
1. Go to Reuters or BBC News (`bbc.com/news`).
2. Download any genuine news photo and copy its exact caption.
3. Click **"Analyze & Verify"**:
   * The system will verify that visual features, text semantics, and physical sensor noise are mutually coherent, classifying it as **`REAL` (Pristine)**.

### Test Case 3: The "Generative AI" Test (Diffusion / Midjourney)
1. Download any image from `reddit.com/r/midjourney` or `lexica.art` (e.g., an AI-generated politician or futuristic cityscape).
2. Write a realistic caption for it.
3. Click **"Analyze & Verify"**:
   * The **Physical Forensic Branch** (Laplacian filter + 2D FFT) will detect the lack of sensor PRNU and the presence of upsampling grid frequencies, accurately flagging it as **`FAKE` (Manipulated / Synthetic Image)**.
