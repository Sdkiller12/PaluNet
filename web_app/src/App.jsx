import React, { useState, createContext, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Analyzer from './pages/Analyzer';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import AdminDashboard from './pages/AdminDashboard';
import { AlertTriangle, ShieldCheck } from 'lucide-react';
import './App.css';

export const AuthContext = createContext();

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(localStorage.getItem('isLoggedIn') === 'true');
  const [userRole, setUserRole] = useState(localStorage.getItem('userRole') || 'researcher');

  const login = (email) => {
    // Si l'email est "admin@palunet.org", on donne le rôle admin.
    const role = email === 'admin@palunet.org' ? 'admin' : 'researcher';
    localStorage.setItem('isLoggedIn', 'true');
    localStorage.setItem('userRole', role);
    setIsAuthenticated(true);
    setUserRole(role);
  };

  const logout = () => {
    localStorage.removeItem('isLoggedIn');
    localStorage.removeItem('userRole');
    setIsAuthenticated(false);
    setUserRole('researcher');
  };

  // Composant pour protéger les routes privées
  const PrivateRoute = ({ children, requiredRole }) => {
    if (!isAuthenticated) return <Navigate to="/login" />;
    if (requiredRole && userRole !== requiredRole) return <Navigate to="/" />;
    return children;
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, userRole, login, logout }}>
      <Router>
        <div className="app-wrapper">
          <div className="disclaimer" role="note">
            <AlertTriangle size={20} />
            <strong>Outil de recherche médicale</strong> — ne remplace pas un diagnostic médical certifié. Version de démonstration.
          </div>
          
          <Navbar />

          <main className="main-content">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/login" element={<Login />} />
              
              {/* Routes protégées (Chercheur et Admin) */}
              <Route path="/analyze" element={<PrivateRoute><Analyzer /></PrivateRoute>} />
              <Route path="/dashboard" element={<PrivateRoute><Dashboard /></PrivateRoute>} />
              
              {/* Route strictement réservée à l'Admin */}
              <Route path="/admin" element={<PrivateRoute requiredRole="admin"><AdminDashboard /></PrivateRoute>} />
            </Routes>
          </main>

          <footer className="app-footer">
            <p>
              <ShieldCheck size={18} />
              Confidentialité garantie. Aucune image n'est conservée. Les journaux sont anonymisés.
            </p>
          </footer>
        </div>
      </Router>
    </AuthContext.Provider>
  );
}

export default App;
