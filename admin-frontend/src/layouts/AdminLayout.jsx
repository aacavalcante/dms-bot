// src/layouts/AdminLayout.jsx
import React, { useState } from 'react';
import { Outlet, NavLink, Navigate, useNavigate } from 'react-router-dom';
import { Navbar, Container, Nav, Button, Form } from 'react-bootstrap';
import { useAuth } from '../auth/AuthContext';
import {
  AiFillHome,
  AiOutlineUnorderedList,
  AiOutlineUsergroupAdd,
  AiOutlineUser,
  AiOutlineSearch,
  AiOutlineClose,
} from 'react-icons/ai';
import logo from '../assets/logo.png';

export default function AdminLayout() {
  const { token, logout } = useAuth();
  const navigate = useNavigate();

  // Hooks devem ficar no topo, sempre
  const [showSearch, setShowSearch] = useState(false);
  const [search, setSearch] = useState('');

  // Se não estiver logado, redireciona
  if (!token) {
    return <Navigate to="/login" replace />;
  }

  // Saudação
  const hour = new Date().getHours();
  const greeting =
    hour < 12 ? 'Bom dia' : hour < 18 ? 'Boa tarde' : 'Boa noite';

  return (
    <div className="d-flex flex-column" style={{ minHeight: '100vh' }}>
      {/* Header */}
      <Navbar bg="dark" variant="dark" className="px-3">
        <Container fluid className="d-flex align-items-center">
          <Navbar.Brand
            as={NavLink}
            to="/"
            className="d-flex align-items-center p-0"
          >
            <img
              src={logo}
              alt="Logo"
              style={{ height: '40px', marginRight: '8px' }}
            />
            <span style={{ fontWeight: 600 }}>Minha Admin</span>
          </Navbar.Brand>

          <div className="ms-auto d-flex align-items-center">
            {showSearch ? (
              <Form className="d-flex align-items-center">
                <Form.Control
                  type="search"
                  placeholder="Buscar..."
                  size="sm"
                  className="me-2"
                  style={{ borderRadius: '20px' }}
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                />
                <Button
                  variant="outline-light"
                  size="sm"
                  onClick={() => setShowSearch(false)}
                  style={{ borderRadius: '50%' }}
                >
                  <AiOutlineClose />
                </Button>
              </Form>
            ) : (
              <Button
                variant="link"
                className="text-light fs-4"
                onClick={() => setShowSearch(true)}
              >
                <AiOutlineSearch />
              </Button>
            )}

            <div className="text-light ms-3 text-end">
              <div style={{ fontWeight: 600 }}>{greeting}</div>
              <small>
                {new Date().toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </small>
            </div>

            <Button
              variant="outline-light"
              size="sm"
              className="ms-3"
              onClick={() => {
                logout();
                navigate('/login');
              }}
            >
              Sair
            </Button>
          </div>
        </Container>
      </Navbar>

      {/* Conteúdo */}
      <Container
        fluid
        style={{ flex: 1, padding: '20px', marginBottom: '60px' }}
      >
        <Outlet />
      </Container>

      {/* Menu Inferior */}
      <Navbar
        bg="light"
        variant="light"
        fixed="bottom"
        className="border-top"
      >
        <Nav className="w-100 justify-content-around">
          <Nav.Link
            as={NavLink}
            to="/"
            className="d-flex flex-column align-items-center text-dark"
          >
            <AiFillHome size={24} />
            <small>Início</small>
          </Nav.Link>

          <Nav.Link
            as={NavLink}
            to="/sessions"
            className="d-flex flex-column align-items-center text-dark"
          >
            <AiOutlineUnorderedList size={24} />
            <small>Sessões</small>
          </Nav.Link>

          <Nav.Link
            as={NavLink}
            to="/leads"
            className="d-flex flex-column align-items-center text-dark"
          >
            <AiOutlineUsergroupAdd size={24} />
            <small>Leads</small>
          </Nav.Link>

          <Nav.Link
            as={NavLink}
            to="/admin-users"
            className="d-flex flex-column align-items-center text-dark"
          >
            <AiOutlineUser size={24} />
            <small>Admins</small>
          </Nav.Link>
        </Nav>
      </Navbar>
    </div>
  );
}
