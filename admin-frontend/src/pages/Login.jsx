import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { login as apiLogin } from '../api/auth';
import { useAuth } from '../auth/AuthContext';
import { Container, Form, Button } from 'react-bootstrap';

export default function Login() {
  const [username, setUser] = useState('');
  const [password, setPass] = useState('');
  const { login } = useAuth();
  const nav = useNavigate();

  const onSubmit = async (e) => {
    e.preventDefault();
    try {
      const token = await apiLogin(username, password);
      login(token);
      nav('/');
    } catch {
      alert('Usuário ou senha inválidos');
    }
  };

  return (
    <Container
      fluid
      className="d-flex flex-column align-items-center justify-content-center"
      style={{ minHeight: '100vh', background: 'var(--color-gray-light)' }}
    >
      <Form
        onSubmit={onSubmit}
        style={{ width: 300, gap: 'var(--spacing-sm)', display: 'grid' }}
      >
        <h2 className="text-center">Admin Login</h2>
        <Form.Group>
          <Form.Label>Usuário</Form.Label>
          <Form.Control
            type="text"
            value={username}
            onChange={(e) => setUser(e.target.value)}
          />
        </Form.Group>
        <Form.Group>
          <Form.Label>Senha</Form.Label>
          <Form.Control
            type="password"
            value={password}
            onChange={(e) => setPass(e.target.value)}
          />
        </Form.Group>
        <Button type="submit" variant="primary" className="mt-3">
          Entrar
        </Button>
      </Form>
    </Container>
  );
}
