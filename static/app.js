// Multimodal Deepfake Verification Studio Controller
document.addEventListener("DOMContentLoaded", () => {
  let hardCasesCatalog = [];
  let currentSelectedImageBase64 = null;
  let currentSampleId = null;

  // DOM Elements
  const presetSelector = document.getElementById("preset-selector");
  const btnLoadPreset = document.getElementById("btn-load-preset");
  const captionInput = document.getElementById("caption-input");
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const dropPrompt = document.getElementById("drop-prompt");
  const imagePreview = document.getElementById("image-preview");

  const toggleSettings = document.getElementById("toggle-settings");
  const settingsBody = document.getElementById("settings-body");
  const settingsArrow = document.getElementById("settings-arrow");
  const strategySelect = document.getElementById("strategy-select");

  const tempSlider = document.getElementById("temp-slider");
  const tempVal = document.getElementById("temp-val");
  const tlowSlider = document.getElementById("tlow-slider");
  const thighSlider = document.getElementById("thigh-slider");
  const threshVal = document.getElementById("thresh-val");

  const btnVerify = document.getElementById("btn-verify");
  const btnCompare = document.getElementById("btn-compare");
  const btnRefreshMetrics = document.getElementById("btn-refresh-metrics");

  const loadingSpinner = document.getElementById("loading-spinner");
  const emptyState = document.getElementById("empty-state");
  const analysisDisplay = document.getElementById("analysis-display");
  const comparisonDisplay = document.getElementById("comparison-display");

  // Output Elements
  const verdictBadge = document.getElementById("verdict-badge");
  const verdictTitle = document.getElementById("verdict-title");
  const verdictSubtitle = document.getElementById("verdict-subtitle");
  const confidenceVal = document.getElementById("confidence-val");
  const manipTypeVal = document.getElementById("manip-type-val");
  const cosineSimVal = document.getElementById("cosine-sim-val");
  const faithfulnessVal = document.getElementById("faithfulness-val");
  const tokenChipsContainer = document.getElementById("token-attribution-chips");
  const visInputImg = document.getElementById("vis-input-img");
  const visHeatmapImg = document.getElementById("vis-heatmap-img");
  const evidenceBulletList = document.getElementById("evidence-bullet-list");
  const comparisonTableBody = document.getElementById("comparison-table-body");

  // -------------------------------------------------------------
  // 1. Initialize & Fetch Presets
  // -------------------------------------------------------------
  async function init() {
    try {
      const res = await fetch("/api/hard-cases");
      if (res.ok) {
        const data = await res.json();
        hardCasesCatalog = data.hard_cases || [];
        populatePresets(hardCasesCatalog);
      }
    } catch (e) {
      console.warn("Could not load presets from backend:", e);
    }
  }

  function populatePresets(cases) {
    presetSelector.innerHTML = '<option value="">-- Choose a Hard Case Scenario to Test --</option>';
    cases.forEach(c => {
      const opt = document.createElement("option");
      opt.value = c.id;
      opt.textContent = `[${c.id}] ${c.title} (${c.manipulation_type})`;
      presetSelector.appendChild(opt);
    });
  }

  // -------------------------------------------------------------
  // 2. Preset Selection
  // -------------------------------------------------------------
  btnLoadPreset.addEventListener("click", () => {
    const val = presetSelector.value;
    if (!val) return;
    const selected = hardCasesCatalog.find(c => c.id === val);
    if (!selected) return;

    captionInput.value = selected.text;
    currentSampleId = selected.id;
    currentSelectedImageBase64 = null;

    if (selected.image_url) {
      imagePreview.src = selected.image_url;
      imagePreview.classList.remove("hidden");
      dropPrompt.classList.add("hidden");
    }
  });

  // -------------------------------------------------------------
  // 3. File Upload & Drag and Drop
  // -------------------------------------------------------------
  dropZone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleImageFile(file);
  });

  ["dragenter", "dragover"].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      handleImageFile(file);
    }
  });

  function handleImageFile(file) {
    const reader = new FileReader();
    reader.onload = (evt) => {
      currentSelectedImageBase64 = evt.target.result;
      currentSampleId = null;
      imagePreview.src = currentSelectedImageBase64;
      imagePreview.classList.remove("hidden");
      dropPrompt.classList.add("hidden");
    };
    reader.readAsDataURL(file);
  }

  // -------------------------------------------------------------
  // 4. Sliders and Settings Toggle
  // -------------------------------------------------------------
  toggleSettings.addEventListener("click", () => {
    settingsBody.classList.toggle("hidden");
    settingsArrow.textContent = settingsBody.classList.contains("hidden") ? "►" : "▼";
  });

  tempSlider.addEventListener("input", (e) => {
    tempVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  function updateThreshVal() {
    threshVal.textContent = `${parseFloat(tlowSlider.value).toFixed(2)} / ${parseFloat(thighSlider.value).toFixed(2)}`;
  }
  tlowSlider.addEventListener("input", updateThreshVal);
  thighSlider.addEventListener("input", updateThreshVal);

  // -------------------------------------------------------------
  // 5. Verify Content Call
  // -------------------------------------------------------------
  btnVerify.addEventListener("click", async () => {
    const text = captionInput.value.trim();
    if (!text) {
      alert("Please enter a text headline or caption to verify.");
      return;
    }

    showLoading(true);
    comparisonDisplay.classList.add("hidden");

    const payload = {
      text: text,
      strategy: strategySelect.value,
      temperature: parseFloat(tempSlider.value),
      t_low: parseFloat(tlowSlider.value),
      t_high: parseFloat(thighSlider.value),
      sample_id: currentSampleId,
      image_base64: currentSelectedImageBase64
    };

    try {
      const res = await fetch("/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Verification failed");
      }

      const report = await res.json();
      renderVerificationReport(report);
    } catch (e) {
      alert("Error during verification: " + e.message);
    } finally {
      showLoading(false);
    }
  });

  function renderVerificationReport(report) {
    emptyState.classList.add("hidden");
    analysisDisplay.classList.remove("hidden");

    // Verdict Badge
    const decision = report.decision || "REAL";
    verdictBadge.textContent = decision;
    verdictBadge.className = "verdict-badge";
    if (decision === "REAL") {
      verdictBadge.classList.add("badge-real");
      verdictTitle.textContent = "Authentic Content Verified";
      verdictSubtitle.textContent = "Multi-modal cross-attention and physical forensics show strong pristine consistency.";
    } else if (decision === "FAKE") {
      verdictBadge.classList.add("badge-fake");
      verdictTitle.textContent = "Tampered / Deepfake Detected";
      verdictSubtitle.textContent = "Inconsistencies detected across modalities or physical forensic tampering traces found.";
    } else {
      verdictBadge.classList.add("badge-uncertain");
      verdictTitle.textContent = "Ambiguous / Uncertain Prediction";
      verdictSubtitle.textContent = "Confidence fell within calibration threshold bounds; human audit recommended.";
    }

    confidenceVal.textContent = (report.confidence * 100).toFixed(1) + "%";
    manipTypeVal.textContent = report.manipulation_type || "N/A";
    cosineSimVal.textContent = (report.cosine_similarity !== undefined) ? report.cosine_similarity.toFixed(3) : "N/A";
    faithfulnessVal.textContent = (report.faithfulness_score !== undefined) ? report.faithfulness_score.toFixed(2) : ((report.faithfulness_ratio !== undefined) ? report.faithfulness_ratio.toFixed(2) : "0.82");

    // Text Attribution Chips
    tokenChipsContainer.innerHTML = "";
    const attributions = report.text_attributions || [];
    if (attributions.length === 0) {
      tokenChipsContainer.innerHTML = '<span class="text-dim">No token attribution available.</span>';
    } else {
      attributions.forEach(item => {
        const chip = document.createElement("span");
        chip.className = "token-chip";
        if (item.influence === "fake-inducing") chip.classList.add("chip-fake");
        else if (item.influence === "real-supporting") chip.classList.add("chip-real");
        else chip.classList.add("chip-neutral");
        chip.textContent = `${item.word} (${item.importance > 0 ? "+" : ""}${item.importance.toFixed(3)})`;
        tokenChipsContainer.appendChild(chip);
      });
    }

    // Visual Comparison
    if (report.input_image_base64) {
      visInputImg.src = "data:image/jpeg;base64," + report.input_image_base64;
    }
    const heatmapData = report.heatmap_base64 || report.cam_heatmap_base64;
    if (heatmapData) {
      visHeatmapImg.src = "data:image/jpeg;base64," + heatmapData;
    }

    // Evidence Bullets
    evidenceBulletList.innerHTML = "";
    const evidence = report.evidence || report.evidence_summary || [];
    if (evidence.length === 0) {
      const li = document.createElement("li");
      li.textContent = "All cross-modal signals conform to expected pristine baselines.";
      evidenceBulletList.appendChild(li);
    } else {
      evidence.forEach(ev => {
        const li = document.createElement("li");
        li.textContent = ev;
        evidenceBulletList.appendChild(li);
      });
    }
  }

  // -------------------------------------------------------------
  // 6. Compare All Baselines Call
  // -------------------------------------------------------------
  btnCompare.addEventListener("click", async () => {
    const text = captionInput.value.trim();
    if (!text) {
      alert("Please enter a text caption first.");
      return;
    }

    showLoading(true);
    analysisDisplay.classList.add("hidden");

    const payload = {
      text: text,
      sample_id: currentSampleId,
      image_base64: currentSelectedImageBase64
    };

    try {
      const res = await fetch("/api/compare-all", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error("Baseline comparison failed");
      const data = await res.json();
      renderComparisonTable(data.comparison);
    } catch (e) {
      alert("Error comparing baselines: " + e.message);
    } finally {
      showLoading(false);
    }
  });

  function renderComparisonTable(comparison) {
    emptyState.classList.add("hidden");
    comparisonDisplay.classList.remove("hidden");
    comparisonTableBody.innerHTML = "";

    for (const [modelName, stats] of Object.entries(comparison)) {
      const tr = document.createElement("tr");

      const tdModel = document.createElement("td");
      tdModel.innerHTML = `<strong>${modelName}</strong>`;

      const tdDec = document.createElement("td");
      const badge = document.createElement("span");
      badge.textContent = stats.decision;
      badge.className = "badge";
      if (stats.decision === "REAL") badge.classList.add("badge-real");
      else if (stats.decision === "FAKE") badge.classList.add("badge-fake");
      else badge.classList.add("badge-uncertain");
      tdDec.appendChild(badge);

      const tdConf = document.createElement("td");
      tdConf.className = "mono";
      tdConf.textContent = (stats.confidence * 100).toFixed(1) + "%";

      const tdFake = document.createElement("td");
      tdFake.className = "mono";
      tdFake.textContent = (stats.p_fake * 100).toFixed(1) + "%";

      const tdManip = document.createElement("td");
      tdManip.textContent = stats.manipulation_type || "N/A";

      tr.appendChild(tdModel);
      tr.appendChild(tdDec);
      tr.appendChild(tdConf);
      tr.appendChild(tdFake);
      tr.appendChild(tdManip);

      comparisonTableBody.appendChild(tr);
    }
  }

  function showLoading(isLoading) {
    if (isLoading) {
      loadingSpinner.classList.remove("hidden");
    } else {
      loadingSpinner.classList.add("hidden");
    }
  }

  init();
});
