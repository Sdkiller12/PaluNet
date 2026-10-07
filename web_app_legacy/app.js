"use strict";

const API_BASE = ""; // même origine que l'API (servie par FastAPI)
const MAX_BYTES = 5 * 1024 * 1024;
const LABELS_FR = { Parasitized: "Parasitée", Uninfected: "Saine" };

const $ = (id) => document.getElementById(id);
const fileInput = $("file");
const apiKeyInput = $("api-key");
let selectedFile = null;

function storage(fn) { try { return fn(window.localStorage); } catch { return null; } }
apiKeyInput.value = storage((s) => s.getItem("palunet_api_key")) || "";
apiKeyInput.addEventListener("change", () => storage((s) => s.setItem("palunet_api_key", apiKeyInput.value)));

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = false;
  $("result").hidden = true;
}

function pct(x) { return (x * 100).toFixed(1).replace(".", ",") + " %"; }

async function checkHealth() {
  const el = $("status");
  try {
    const r = await fetch(API_BASE + "/health");
    const h = await r.json();
    if (!r.ok) throw new Error(h.detail || "indisponible");
    el.textContent = `API opérationnelle — modèle v${h.model_version}, seuil de décision ${h.decision_threshold}`;
    el.className = "status ok";
    if (!h.heatmap_available) $("heatmap").disabled = true;
  } catch (e) {
    el.textContent = "API indisponible : " + e.message;
    el.className = "status ko";
  }
}

function selectFile(file) {
  $("error").hidden = true;
  if (!file) return;
  if (!["image/png", "image/jpeg"].includes(file.type)) {
    showError("Format non supporté : seules les images PNG et JPEG sont acceptées.");
    return;
  }
  if (file.size > MAX_BYTES) {
    showError("Fichier trop volumineux : la taille maximale est de 5 Mo.");
    return;
  }
  selectedFile = file;
  $("dropzone-text").textContent = file.name;
  $("submit").disabled = false;
}

// D3 : messages d'erreur compréhensibles
function describeError(status, detail) {
  switch (status) {
    case 400: return "Image refusée : " + (detail || "fichier invalide ou corrompu.");
    case 401: return "Accès refusé : clé API manquante ou invalide (voir « Clé API »).";
    case 413: return "Fichier trop volumineux : la taille maximale est de 5 Mo.";
    case 429: return "Trop de requêtes : merci de patienter une minute avant de réessayer.";
    case 503: return "Le service d'analyse n'est pas prêt (modèle non chargé). Réessayez plus tard.";
    default:  return "Erreur inattendue du serveur (" + status + "). Réessayez plus tard.";
  }
}

function render(result) {
  const positive = result.prediction === "Parasitized";
  const name = LABELS_FR[result.prediction] || result.prediction;
  const pred = $("prediction");
  pred.textContent = name;
  pred.className = "prediction " + (positive ? "positive" : "negative");
  $("confidence-label").textContent = `Confiance que la cellule est « ${name.toLowerCase()} »`;
  $("confidence").textContent = pct(result.confidence);
  $("confidence-bar").style.width = (result.confidence * 100) + "%";
  $("proba").textContent = "Probabilité d'infection (parasitée) : " + pct(result.probability_parasitized);
  $("meta").textContent = `Modèle v${result.model_version} — ${result.processing_time_ms} ms`;

  const warnings = $("warnings");
  warnings.replaceChildren(...(result.quality_warnings || []).map((w) => {
    const li = document.createElement("li");
    li.textContent = "Qualité d'image : " + w;
    return li;
  }));

  const overlay = $("overlay");
  if (result.heatmap_base64) {
    overlay.src = "data:image/png;base64," + result.heatmap_base64;
    overlay.hidden = false;
  } else {
    overlay.hidden = true;
  }
  $("preview").src = URL.createObjectURL(selectedFile);
  $("error").hidden = true;
  $("result").hidden = false;
}

async function submit(event) {
  event.preventDefault();
  if (!selectedFile) return;
  const body = new FormData();
  body.append("file", selectedFile);
  body.append("include_heatmap", $("heatmap").checked ? "true" : "false");
  const headers = apiKeyInput.value ? { "X-API-Key": apiKeyInput.value } : {};
  const button = $("submit");
  button.disabled = true;
  button.textContent = "Analyse en cours…";
  try {
    const r = await fetch(API_BASE + "/v1/predict", { method: "POST", body, headers });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      showError(describeError(r.status, typeof data.detail === "string" ? data.detail : ""));
      return;
    }
    render(data);
  } catch {
    showError("Impossible de joindre l'API. Vérifiez votre connexion.");
  } finally {
    button.disabled = false;
    button.textContent = "Analyser";
  }
}

const dropzone = $("dropzone");
fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  selectFile(e.dataTransfer.files[0]);
});
$("form").addEventListener("submit", submit);
checkHealth();
