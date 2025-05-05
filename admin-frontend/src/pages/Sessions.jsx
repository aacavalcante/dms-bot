// src/pages/Sessions.jsx
import React, { useEffect, useState } from 'react';
import { useAuth } from '../auth/AuthContext';
import { Container, Card, Table, Spinner } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';

export default function Sessions() {
  const [sessions, setSessions] = useState(null);
  const { token } = useAuth();
  const navigate = useNavigate(); // hook para navegação

  useEffect(() => {
    fetch('/api/sessions', {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(res => res.json())
      .then(setSessions)
      .catch(console.error);
  }, [token]);

  return (
    <Container className="py-4">
      <h2>Lista de Sessões</h2>

      {sessions === null ? (
        <div className="text-center py-5">
          <Spinner animation="border" size="sm" />
        </div>
      ) : sessions.length === 0 ? (
        <p className="small">Nenhuma sessão encontrada.</p>
      ) : (
        <Card className="shadow-sm w-100 mb-4">
          <Card.Body className="p-0">
            <Table hover size="sm" className="mb-0 small">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Telefone</th>
                  <th>Início</th>
                </tr>
              </thead>
              <tbody>
                {sessions.map(s => (
                  <tr
                    key={s.id}
                    onClick={() => navigate(`/sessions/${s.id}`)}
                    style={{ cursor: 'pointer' }}
                  >
                    <td>{s.id}</td>
                    <td>{s.userPhone}</td>
                    <td>{new Date(s.startedAt).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </Card.Body>
        </Card>
      )}
    </Container>
  );
}
