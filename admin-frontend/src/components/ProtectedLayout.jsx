import React from 'react';
import { Outlet, Navigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import TopBar from './TopBar';
import NavBar from './NavBar';

export default function ProtectedLayout() {
  const { token } = useAuth();
  if (!token) return <Navigate to="/login" replace />;

  return (
    <div className="d-flex flex-column vh-100">
      <TopBar />
      <div className="flex-grow-1 overflow-auto">
        <Outlet />
      </div>
      <NavBar />
    </div>
  );
}
