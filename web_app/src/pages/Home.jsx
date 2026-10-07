import React, { useContext } from 'react';
import { Link } from 'react-router-dom';
import { Activity, ShieldCheck, Microscope, Database, ArrowRight, Lock, Scan, CheckCircle2 } from 'lucide-react';
import { AuthContext } from '../App';

export default function Home() {
  const { isAuthenticated } = useContext(AuthContext);

  return (
    <div className="page-container animate-fade-in">
      {/* Hero Section */}
      <section className="hero-section hero-layout" style={{ padding: '40px 0 60px' }}>
        <div className="hero-content">
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '6px 12px', background: 'rgba(14,165,233,0.1)', color: 'var(--primary)', borderRadius: '20px', fontSize: '0.85rem', fontWeight: '600', marginBottom: '24px' }}>
            <span className="pulse" style={{ width: '8px', height: '8px', background: 'var(--primary)', borderRadius: '50%', display: 'inline-block' }}></span>
            Plateforme IA Nouvelle Génération
          </div>
          <h1 style={{ textAlign: 'left', marginBottom: '20px', fontSize: '2.8rem' }}>L'IA au service de la lutte contre le Paludisme</h1>
          <p className="hero-subtitle" style={{ textAlign: 'left', marginLeft: '0' }}>
            PaluNet Research automatise la détection de <i>Plasmodium falciparum</i> sur les frottis sanguins. Une aide précieuse pour les laboratoires et le diagnostic en milieu endémique.
          </p>
          <div className="hero-actions" style={{ justifyContent: 'flex-start' }}>
            {isAuthenticated ? (
              <Link to="/analyze" className="btn-primary" style={{ display: 'inline-flex', width: 'auto', padding: '14px 32px', textDecoration: 'none' }}>
                Démarrer une analyse <ArrowRight size={20} />
              </Link>
            ) : (
              <Link to="/login" className="btn-primary" style={{ display: 'inline-flex', width: 'auto', padding: '14px 32px', textDecoration: 'none' }}>
                <Lock size={20} /> Connexion requise
              </Link>
            )}
          </div>
        </div>
        
        {/* Animated Visual on the right */}
        <div className="hero-visual">
          {/* Glowing background */}
          <div className="visual-circle"></div>
          
          {/* Floating UI Cards */}
          <div className="glass-card floating" style={{ top: '10%', right: '5%', zIndex: 2 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{ background: 'var(--negative)', color: 'white', padding: '10px', borderRadius: '12px' }}>
                <CheckCircle2 size={24} />
              </div>
              <div>
                <p style={{ margin: 0, fontWeight: '800', fontSize: '1.2rem', color: 'var(--text)' }}>96.8%</p>
                <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--muted)', fontWeight: '600' }}>Précision IA</p>
              </div>
            </div>
          </div>

          <div className="glass-card floating-delayed" style={{ bottom: '15%', left: '5%', zIndex: 3, padding: '16px' }}>
            <div style={{ position: 'relative', width: '130px', height: '130px', background: 'var(--bg)', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid var(--border)', overflow: 'hidden' }}>
              <img src="https://images.unsplash.com/photo-1530026405186-ed1f139313f8?auto=format&fit=crop&w=200&q=80" alt="Blood cells" style={{ opacity: 0.6, width: '100%', height: '100%', objectFit: 'cover' }} />
              <Scan size={48} color="var(--primary)" style={{ position: 'absolute' }} className="pulse" />
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="features-grid">
        <div className="feature-card animate-slide-down" style={{ animationDelay: '0.1s' }}>
          <Microscope size={40} className="feature-icon" />
          <h3>Analyse Cellulaire Précise</h3>
          <p>Nos modèles identifient la présence de parasites sur cellules uniques avec une précision de pointe.</p>
        </div>
        <div className="feature-card animate-slide-down" style={{ animationDelay: '0.2s' }}>
          <Activity size={40} className="feature-icon" />
          <h3>Explicabilité (Grad-CAM)</h3>
          <p>L'IA surligne les zones de la cellule l'ayant aidée à prendre sa décision (Heatmap).</p>
        </div>
        <div className="feature-card animate-slide-down" style={{ animationDelay: '0.3s' }}>
          <Database size={40} className="feature-icon" />
          <h3>Anonymisation Garantie</h3>
          <p>Les images traitées ne sont pas stockées. Seules les métadonnées sont conservées.</p>
        </div>
      </section>

      {/* About Section */}
      <section className="about-section card animate-slide-down" style={{ marginTop: '64px', overflow: 'hidden', animationDelay: '0.4s' }}>
        <div className="about-layout">
          <div style={{ flex: 1 }}>
            <h2>Pourquoi PaluNet ?</h2>
            <p style={{ marginTop: '16px', lineHeight: '1.8' }}>
              Le diagnostic manuel du paludisme au microscope est une tâche fastidieuse, chronophage et qui nécessite une grande expertise. Dans les zones endémiques, le manque de personnel qualifié peut entraîner des retards cruciaux.
              <br/><br/>
              PaluNet offre un <strong>second avis numérique instantané</strong>. Grâce à une API robuste et une interface intuitive, analysez une image de cellule sanguine en quelques millisecondes pour épauler les laboratoires et les cliniciens.
            </p>
          </div>
          <img src="https://images.unsplash.com/photo-1579684385127-1ef15d508118?auto=format&fit=crop&w=600&q=80" alt="Laboratory" className="about-image" />
        </div>
      </section>
    </div>
  );
}
