import React, { useEffect, useState } from 'react';
import { useAuth } from '../auth/AuthContext';
import { Container, Card, Table, Spinner } from 'react-bootstrap';

export default function Leads() {
  const [leads, setLeads] = useState(null);
  const { token } = useAuth();

  useEffect(() => {
    fetch('/api/leads', {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then(setLeads)
      .catch(console.error);
  }, [token]);

  return (
    <Container className="py-4">
      <h2>Leads</h2>

      {leads === null ? (
        <Spinner animation="border" size="sm" />
      ) : leads.length === 0 ? (
        <p className="small">Nenhum lead encontrado.</p>
      ) : (
        // Card ocupa 100% da largura do container, alinhado com o h2
        <Card className="shadow-sm w-100 mb-4">
          <Card.Body className="p-0">
            <Table hover size="sm" className="mb-0 small">
              <thead>
                <tr>
                  <th>Telefone</th>
                  <th>Nome</th>
                  <th>E-mail</th>
                </tr>
              </thead>
              <tbody>
                {leads.map((l) => (
                  <tr key={l.phone}>
                    <td>{l.phone}</td>
                    <td>{l.name || '—'}</td>
                    <td>{l.email || '—'}</td>
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
