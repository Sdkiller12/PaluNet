import React, { useContext } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, Home, FlaskConical, LayoutDashboard, LogIn, LogOut, ShieldAlert } from 'lucide-react';
import { AuthContext } from '../App';

export default function Navbar() {
  const location = useLocation();
  const { isAuthenticated, userRole, logout } = useContext(AuthContext);
  
  const isActive = (path) => location.pathname === path ? 'active' : '';

  return (
    <nav className="topbar">
      <Link to="/" className="brand" style={{ textDecoration: 'none' }}>
        <Activity size={28} />
        PaluNet Research
      </Link>
      
      <div className="nav-links">
        <Link to="/" className={`nav-link ${isActive('/')}`}><Home size={18}/> Accueil</Link>
        {isAuthenticated && (
          <>
            <Link to="/analyze" className={`nav-link ${isActive('/analyze')}`}><FlaskConical size={18}/> Analyser</Link>
            <Link to="/dashboard" className={`nav-link ${isActive('/dashboard')}`}><LayoutDashboard size={18}/> Tableau de bord</Link>
          </>
        )}
        {isAuthenticated && userRole === 'admin' && (
          <Link to="/admin" className={`nav-link ${isActive('/admin')}`}><ShieldAlert size={18}/> Administration</Link>
        )}
      </div>

      <div className="nav-actions">
        {isAuthenticated ? (
          <button onClick={logout} className="btn-secondary" style={{ cursor: 'pointer', outline: 'none' }}>
            <LogOut size={18}/> Déconnexion
          </button>
        ) : (
          <Link to="/login" className="btn-secondary" style={{ textDecoration: 'none' }}>
            <LogIn size={18}/> Connexion
          </Link>
        )}
      </div>
    </nav>
  );
}
