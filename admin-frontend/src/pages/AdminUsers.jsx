import React, { useEffect, useState } from 'react';
import { useAuth } from '../auth/AuthContext';
import { Container, Card, ListGroup, Spinner } from 'react-bootstrap';

export default function AdminUsers() {
  const [users, setUsers] = useState(null);
  const { token } = useAuth();

  useEffect(() => {
    fetch('/api/users', {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then(setUsers)
      .catch(console.error);
  }, [token]);

  return (
    <Container className="py-4">
      <h2>Usuários Administrativos</h2>

      {users === null ? (
        <Spinner animation="border" size="sm" />
      ) : users.length === 0 ? (
        <p className="small">Nenhum usuário cadastrado.</p>
      ) : (
        <Card className="shadow-sm w-100 mb-4">
          <ListGroup variant="flush" className="small">
            {users.map((u) => (
              <ListGroup.Item
                key={u.id}
                className="d-flex justify-content-between p-2"
              >
                <span>{u.username}</span>
                <span className="text-muted">{u.role}</span>
              </ListGroup.Item>
            ))}
          </ListGroup>
        </Card>
      )}
    </Container>
  );
}
