import React from 'react';
import { BarChart3, Activity, Users, FileImage } from 'lucide-react';

export default function Dashboard() {
  return (
    <div className="page-container animate-slide-down">
      <div className="header-section" style={{ textAlign: 'left', marginBottom: '32px' }}>
        <h1 style={{ fontSize: '2rem' }}>Tableau de bord Global</h1>
        <p>Aperçu des performances du modèle et de l'utilisation de l'API (Mode Démo).</p>
      </div>

      <div className="stats-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '24px', marginBottom: '32px' }}>
        <div className="card stat-card" style={{ padding: '24px', marginBottom: '0' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <span style={{ fontWeight: '600', color: 'var(--muted)' }}>Analyses (Aujourd'hui)</span>
            <Activity size={24} style={{ color: 'var(--primary)' }} />
          </div>
          <p style={{ fontSize: '2rem', fontWeight: '800', margin: '0 0 8px 0' }}>1 432</p>
          <span style={{ color: 'var(--negative)', fontSize: '0.85rem', fontWeight: '500' }}>+12% vs hier</span>
        </div>
        
        <div className="card stat-card" style={{ padding: '24px', marginBottom: '0' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <span style={{ fontWeight: '600', color: 'var(--muted)' }}>Précision globale</span>
            <BarChart3 size={24} style={{ color: 'var(--negative)' }} />
          </div>
          <p style={{ fontSize: '2rem', fontWeight: '800', margin: '0 0 8px 0' }}>96.8%</p>
          <span style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>Sur 27,000 images testées</span>
        </div>

        <div className="card stat-card" style={{ padding: '24px', marginBottom: '0' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <span style={{ fontWeight: '600', color: 'var(--muted)' }}>Chercheurs Actifs</span>
            <Users size={24} style={{ color: 'var(--primary-dark)' }} />
          </div>
          <p style={{ fontSize: '2rem', fontWeight: '800', margin: '0 0 8px 0' }}>45</p>
          <span style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>Réseau international</span>
        </div>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: '20px' }}>Historique récent (Anonymisé)</h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid var(--border)' }}>
                <th style={{ padding: '12px', color: 'var(--muted)' }}>ID Requête</th>
                <th style={{ padding: '12px', color: 'var(--muted)' }}>Heure</th>
                <th style={{ padding: '12px', color: 'var(--muted)' }}>Résultat</th>
                <th style={{ padding: '12px', color: 'var(--muted)' }}>Confiance</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '12px', fontWeight: '500' }}>#REQ-8374</td>
                <td style={{ padding: '12px', color: 'var(--muted)' }}>Il y a 5 min</td>
                <td style={{ padding: '12px' }}><span style={{ padding: '4px 8px', borderRadius: '4px', background: 'var(--negative)', color: '#fff', fontSize: '0.8rem', fontWeight: '600' }}>Saine</span></td>
                <td style={{ padding: '12px', fontWeight: '600' }}>99.2%</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '12px', fontWeight: '500' }}>#REQ-8373</td>
                <td style={{ padding: '12px', color: 'var(--muted)' }}>Il y a 12 min</td>
                <td style={{ padding: '12px' }}><span style={{ padding: '4px 8px', borderRadius: '4px', background: 'var(--positive)', color: '#fff', fontSize: '0.8rem', fontWeight: '600' }}>Parasitée</span></td>
                <td style={{ padding: '12px', fontWeight: '600' }}>94.7%</td>
              </tr>
              <tr>
                <td style={{ padding: '12px', fontWeight: '500' }}>#REQ-8372</td>
                <td style={{ padding: '12px', color: 'var(--muted)' }}>Il y a 22 min</td>
                <td style={{ padding: '12px' }}><span style={{ padding: '4px 8px', borderRadius: '4px', background: 'var(--negative)', color: '#fff', fontSize: '0.8rem', fontWeight: '600' }}>Saine</span></td>
                <td style={{ padding: '12px', fontWeight: '600' }}>88.1%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
