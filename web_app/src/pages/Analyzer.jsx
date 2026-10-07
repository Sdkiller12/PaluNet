import React, { useState, useRef, useEffect } from 'react';
import { Activity, UploadCloud, AlertCircle } from 'lucide-react';

export default function Analyzer() {
  const [health, setHealth] = useState({ status: 'loading', data: null });
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [heatmap, setHeatmap] = useState(false);
  const [apiKey, setApiKey] = useState(localStorage.getItem('palunet_api_key') || '');
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchHealth();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/health');
      if (!res.ok) throw new Error('API indisponible');
      const data = await res.json();
      setHealth({ status: 'ok', data });
    } catch (err) {
      setHealth({ status: 'error', data: null });
    }
  };

  const handleApiKeyChange = (e) => {
    const val = e.target.value;
    setApiKey(val);
    localStorage.setItem('palunet_api_key', val);
  };

  const onFileSelect = (selectedFile) => {
    setError('');
    setResult(null);
    if (!selectedFile) return;
    if (!['image/png', 'image/jpeg'].includes(selectedFile.type)) {
      setError("Format non supporté : seules les images PNG et JPEG sont acceptées.");
      return;
    }
    if (selectedFile.size > 5 * 1024 * 1024) {
      setError("Fichier trop volumineux : la taille maximale est de 5 Mo.");
      return;
    }
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setIsLoading(true);
    setError('');
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('include_heatmap', heatmap ? 'true' : 'false');

    const headers = {};
    if (apiKey) headers['X-API-Key'] = apiKey;

    try {
      const res = await fetch('/v1/predict', {
        method: 'POST',
        headers,
        body: formData,
      });

      const data = await res.json().catch(() => ({}));
      
      if (!res.ok) {
        let errMsg = "Erreur inattendue.";
        if (res.status === 400) errMsg = "Image refusée : " + (data.detail || "fichier invalide.");
        else if (res.status === 401) errMsg = "Accès refusé : clé API manquante ou invalide.";
        else if (res.status === 413) errMsg = "Fichier trop volumineux.";
        else if (res.status === 429) errMsg = "Trop de requêtes, veuillez patienter.";
        else if (res.status === 503) errMsg = "Service non prêt. Modèle non chargé.";
        throw new Error(errMsg);
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Impossible de joindre l'API. Vérifiez votre connexion.");
    } finally {
      setIsLoading(false);
    }
  };

  const formatPct = (val) => (val * 100).toFixed(1).replace('.', ',') + ' %';
  const LABELS_FR = { Parasitized: "Parasitée", Uninfected: "Saine" };

  return (
    <div className="page-container animate-fade-in">
      <div className="status-badge" style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '6px 12px', background: 'var(--surface)', borderRadius: '20px', border: '1px solid var(--border)', fontSize: '0.85rem', marginBottom: '24px' }}>
        {health.status === 'loading' && <span style={{color: 'var(--muted)'}}>Connexion à l'API…</span>}
        {health.status === 'ok' && (
          <><span style={{color: 'var(--negative)'}}>●</span> API opérationnelle {health.data?.model_version ? 'v' + health.data.model_version : ''}</>
        )}
        {health.status === 'error' && (
          <><span style={{color: 'var(--positive)'}}>●</span> API indisponible</>
        )}
      </div>

      <div className="header-section">
        <h1>Analyse Cellulaire</h1>
        <p>Déposez une image de frottis sanguin pour une prédiction instantanée.</p>
      </div>

      <section className="card animate-fade-in" style={{ animationDelay: '0.1s' }}>
        <form onSubmit={handleSubmit}>
          <div 
            className={`dropzone ${isDragging ? 'drag' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={(e) => onFileSelect(e.target.files[0])} 
              accept="image/png,image/jpeg" 
              hidden 
            />
            <UploadCloud size={48} className="dropzone-icon" />
            <span className="dropzone-text">{file ? file.name : "Déposez une image PNG ou JPEG"}</span>
            <span className="dropzone-sub">Cellule sanguine unique, taille max 5 Mo. Cliquez ou glissez-déposez.</span>
          </div>

          <div className="options-group">
            <label className="checkbox-label">
              <input 
                type="checkbox" 
                checked={heatmap} 
                onChange={(e) => setHeatmap(e.target.checked)}
                disabled={health.data && !health.data.heatmap_available}
              /> 
              Afficher l'explicabilité de l'IA (Grad-CAM)
            </label>
            <details>
              <summary>Paramètres avancés / Clé API</summary>
              <input 
                type="password" 
                placeholder="X-API-Key (si l'API l'exige)" 
                value={apiKey}
                onChange={handleApiKeyChange}
                autoComplete="off"
              />
            </details>
          </div>

          <button type="submit" className="btn-primary" disabled={!file || isLoading}>
            <Activity size={20} />
            {isLoading ? "Analyse en cours..." : "Lancer l'Analyse"}
          </button>
        </form>
      </section>

      {error && (
        <div className="error-banner animate-fade-in">
          <AlertCircle size={24} />
          <span>{error}</span>
        </div>
      )}

      {result && (
        <section className="card result-grid animate-slide-down">
          <div className="preview-container">
            <div className="preview">
              {preview && <img src={preview} alt="Image analysée" />}
              {result.heatmap_base64 && (
                <img src={`data:image/png;base64,${result.heatmap_base64}`} alt="Carte de chaleur Grad-CAM" style={{ opacity: 0.7 }} />
              )}
            </div>
          </div>
          
          <div className="verdict">
            <div className="verdict-header">
              <p className="label">Résultat de l'analyse</p>
              <p className={`prediction ${result.prediction === 'Parasitized' ? 'positive' : 'negative'}`}>
                {LABELS_FR[result.prediction] || result.prediction}
              </p>
            </div>
            
            <div className="confidence-section">
              <p className="label">Niveau de confiance</p>
              <div className="bar-bg">
                <div className="bar-fill" style={{ width: `${result.confidence * 100}%` }}></div>
              </div>
              <p className="confidence">{formatPct(result.confidence)}</p>
              <p className="detail">Probabilité d'infection : {formatPct(result.probability_parasitized)}</p>
            </div>
            
            {result.quality_warnings?.length > 0 && (
              <ul className="warnings-list">
                {result.quality_warnings.map((w, i) => (
                  <li key={i}>Qualité d'image : {w}</li>
                ))}
              </ul>
            )}
            
            <p className="detail" style={{ marginTop: '16px' }}>Modèle v{result.model_version} — {result.processing_time_ms} ms</p>
          </div>
        </section>
      )}
    </div>
  );
}
