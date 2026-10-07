import React, { useState, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogIn, User, Lock, ArrowLeft } from 'lucide-react';
import { AuthContext } from '../App';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();
  const { login } = useContext(AuthContext);

  const handleLogin = (e) => {
    e.preventDefault();
    // Appel du hook de connexion en lui passant l'email
    // Si l'email est admin@palunet.org, l'utilisateur aura les droits administrateur
    login(email);

    if (email === 'admin@palunet.org') {
      navigate('/admin');
    } else {
      navigate('/analyze');
    }
  };

  return (
    <div className="page-container animate-fade-in" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
      <div className="card login-card" style={{ maxWidth: '400px', width: '100%', margin: '0' }}>
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <h2>Espace Chercheur & Admin</h2>
          <p className="muted" style={{ marginTop: '8px', fontSize: '0.9rem', color: 'var(--muted)' }}>
            Connectez-vous pour accéder à l'API. <br />
            <i>(Astuce: Utilisez <b>admin@palunet.org</b> pour tester l'interface Administrateur)</i>
          </p>
        </div>

        <form onSubmit={handleLogin} className="login-form">
          <div className="input-group" style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '0.9rem' }}>Email institutionnel</label>
            <div className="input-wrapper" style={{ position: 'relative' }}>
              <User size={18} style={{ position: 'absolute', left: '12px', top: '12px', color: 'var(--muted)' }} />
              <input 
                type="email" 
                value={email} 
                onChange={e => setEmail(e.target.value)} 
                required 
                placeholder="chercheur@institut.org" 
                style={{ width: '100%', padding: '10px 10px 10px 38px', borderRadius: '8px', border: '1px solid var(--border)' }}
              />
            </div>
          </div>
          <div className="input-group" style={{ marginBottom: '24px' }}>
            <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '0.9rem' }}>Mot de passe</label>
            <div className="input-wrapper" style={{ position: 'relative' }}>
              <Lock size={18} style={{ position: 'absolute', left: '12px', top: '12px', color: 'var(--muted)' }} />
              <input 
                type="password" 
                value={password} 
                onChange={e => setPassword(e.target.value)} 
                required 
                placeholder="••••••••" 
                style={{ width: '100%', padding: '10px 10px 10px 38px', borderRadius: '8px', border: '1px solid var(--border)' }}
              />
            </div>
          </div>
          
          <button type="submit" className="btn-primary" style={{ width: '100%' }}>
            <LogIn size={20} /> Se connecter
          </button>
        </form>

        <div style={{ marginTop: '24px', textAlign: 'center' }}>
          <button onClick={() => navigate(-1)} style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', border: 'none', background: 'none', color: 'var(--muted)', cursor: 'pointer', fontSize: '0.9rem' }}>
            <ArrowLeft size={16} /> Retour
          </button>
        </div>
      </div>
    </div>
  );
}
