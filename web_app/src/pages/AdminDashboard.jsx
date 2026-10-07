import React, { useState } from 'react';
import { ShieldAlert, Plus, Key, Building2, Trash2, Copy, CheckCircle2 } from 'lucide-react';

export default function AdminDashboard() {
  const [institutions, setInstitutions] = useState([
    { id: 1, name: 'Institut Pasteur', email: 'labo@pasteur.fr', apiKey: 'sk_test_pasteur123', status: 'Actif' },
    { id: 2, name: 'CHU de Dakar', email: 'recherche@chu-dakar.sn', apiKey: 'sk_test_chudakar456', status: 'Actif' }
  ]);

  const [newInstName, setNewInstName] = useState('');
  const [newInstEmail, setNewInstEmail] = useState('');
  const [copiedId, setCopiedId] = useState(null);

  const generateApiKey = () => {
    return 'sk_live_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
  };

  const handleAddInstitution = (e) => {
    e.preventDefault();
    if (!newInstName || !newInstEmail) return;

    const newInst = {
      id: Date.now(),
      name: newInstName,
      email: newInstEmail,
      apiKey: generateApiKey(),
      status: 'Actif'
    };

    setInstitutions([...institutions, newInst]);
    setNewInstName('');
    setNewInstEmail('');
  };

  const handleDelete = (id) => {
    setInstitutions(institutions.filter(inst => inst.id !== id));
  };

  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="page-container animate-fade-in">
      <div className="header-section" style={{ textAlign: 'left', marginBottom: '32px' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '2rem' }}>
          <ShieldAlert size={32} color="var(--primary)" />
          Administration
        </h1>
        <p>Gestion des accès institutionnels et des clés API.</p>
      </div>

      <div className="result-grid" style={{ gridTemplateColumns: '1fr 350px' }}>
        
        {/* Liste des institutions */}
        <div className="card" style={{ marginBottom: 0 }}>
          <h3 style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Building2 size={20} /> Accès Partenaires
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid var(--border)' }}>
                  <th style={{ padding: '12px', color: 'var(--muted)' }}>Institut</th>
                  <th style={{ padding: '12px', color: 'var(--muted)' }}>Email (Identifiant)</th>
                  <th style={{ padding: '12px', color: 'var(--muted)' }}>Clé API</th>
                  <th style={{ padding: '12px', color: 'var(--muted)', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {institutions.length === 0 ? (
                  <tr>
                    <td colSpan="4" style={{ padding: '24px', textAlign: 'center', color: 'var(--muted)' }}>Aucune institution enregistrée.</td>
                  </tr>
                ) : (
                  institutions.map(inst => (
                    <tr key={inst.id} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '12px', fontWeight: '500' }}>{inst.name}</td>
                      <td style={{ padding: '12px', color: 'var(--muted)' }}>{inst.email}</td>
                      <td style={{ padding: '12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <code style={{ background: 'var(--bg)', padding: '4px 8px', borderRadius: '4px', fontSize: '0.85rem' }}>
                            {inst.apiKey.substring(0, 8)}...
                          </code>
                          <button onClick={() => copyToClipboard(inst.apiKey, inst.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--muted)' }} title="Copier la clé">
                            {copiedId === inst.id ? <CheckCircle2 size={16} color="var(--negative)" /> : <Copy size={16} />}
                          </button>
                        </div>
                      </td>
                      <td style={{ padding: '12px', textAlign: 'right' }}>
                        <button onClick={() => handleDelete(inst.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--positive)' }} title="Révoquer l'accès">
                          <Trash2 size={18} />
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Formulaire d'ajout */}
        <div className="card" style={{ marginBottom: 0, height: 'fit-content' }}>
          <h3 style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Key size={20} /> Nouvel Accès
          </h3>
          <form onSubmit={handleAddInstitution}>
            <div className="input-group" style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '0.9rem' }}>Nom de l'institut</label>
              <input 
                type="text" 
                value={newInstName} 
                onChange={e => setNewInstName(e.target.value)} 
                required 
                placeholder="Ex: Hôpital Universitaire..." 
                style={{ width: '100%', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border)' }}
              />
            </div>
            <div className="input-group" style={{ marginBottom: '24px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '0.9rem' }}>Email du responsable</label>
              <input 
                type="email" 
                value={newInstEmail} 
                onChange={e => setNewInstEmail(e.target.value)} 
                required 
                placeholder="contact@hopital.org" 
                style={{ width: '100%', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border)' }}
              />
            </div>
            
            <button type="submit" className="btn-primary" style={{ width: '100%', background: 'var(--primary-dark)' }}>
              <Plus size={20} /> Générer les identifiants
            </button>
          </form>
          <div style={{ marginTop: '16px', fontSize: '0.85rem', color: 'var(--muted)', textAlign: 'center' }}>
            Un mot de passe par défaut et une clé API seront générés.
          </div>
        </div>

      </div>
    </div>
  );
}
