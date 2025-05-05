import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Login         from './pages/Login';
import Dashboard     from './pages/Dashboard';
import Sessions      from './pages/Sessions';
import SessionDetail from './pages/SessionDetail';
import Leads         from './pages/Leads';
import AdminUsers    from './pages/AdminUsers';
import ProtectedLayout from './components/ProtectedLayout';


export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />

      <Route element={<ProtectedLayout />}>
        
        <Route path="/"            element={<Dashboard />} />
        <Route path="/sessions"    element={<Sessions />} />
        <Route path="/sessions/:id" element={<SessionDetail />} />
        <Route path="/leads"       element={<Leads />} />
        <Route path="/admin-users" element={<AdminUsers />} />
      </Route>

      <Route path="*" element={<Navigate to="/login" replace />} />
    </Routes>
  );
}
