// src/pages/SessionDetail.jsx
import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import {
  Container,
  Card,
  Table,
  Spinner,
  Button,
  Row,
  Col
} from 'react-bootstrap';
import { ArrowLeft } from 'react-bootstrap-icons'; // Importe o ícone de seta esquerda

function formatDateTime(iso) {
  if (!iso) return '—';
  const dt = new Date(iso);
  return isNaN(dt.getTime())
    ? iso
    : dt.toLocaleString([], {
        hour: '2-digit',
        minute: '2-digit',
      });
}

export default function SessionDetail() {
  const { id } = useParams();
  const [session, setSession] = useState(null);
  const [logs, setLogs] = useState(null);
  const { token } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    fetch(`/api/sessions/${id}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => r.json())
      .then(setSession);

    fetch(`/api/logs/session/${id}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => r.json())
      .then(setLogs);
  }, [id, token]);

  if (!session || logs === null) {
    return (
      <Container className="py-4 text-center">
        <Spinner animation="border" size="sm" /> Carregando…
      </Container>
    );
  }

  const { userPhone, flow, startedAt, endedAt, sessionData } = session;
  const viagem = sessionData?.viagem || {};

  return (
    <Container className="py-4">
      <Button variant="link" onClick={() => navigate(-1)} className="mb-3">
        <div className="d-inline-flex align-items-center">
          <div
            className="rounded-circle bg-light border me-2 d-flex align-items-center justify-content-center"
            style={{ width: '24px', height: '24px' }}
          >
            <ArrowLeft size={16} />
          </div>
          Voltar
        </div>
      </Button>

      {/* Sessão Detalhes */}
      <Card className="shadow-sm w-100 mb-4">
        <Card.Header>Detalhes da Sessão #{id}</Card.Header>
        <Card.Body className="small">
          <Table borderless size="sm" className="mb-3">
            <tbody>
              <tr>
                <th style={{ width: '30%' }}>Telefone</th>
                <td>{userPhone}</td>
              </tr>
              <tr>
                <th>Fluxo</th>
                <td>{flow}</td>
              </tr>
              <tr>
                <th>Início</th>
                <td>{formatDateTime(startedAt)}</td>
              </tr>
              <tr>
                <th>Término</th>
                <td>{formatDateTime(endedAt)}</td>
              </tr>
            </tbody>
          </Table>

          {viagem.destino && (
            <>
              <h5 className="h6">Viagem</h5>
              <Table borderless size="sm" className="small mb-0">
                <tbody>
                  <tr>
                    <th style={{ width: '30%' }}>Destino</th>
                    <td>{viagem.destino}</td>
                  </tr>
                  <tr>
                    <th>Data Início</th>
                    <td>{formatDateTime(viagem.data_inicio)}</td>
                  </tr>
                  <tr>
                    <th>Data Fim</th>
                    <td>{formatDateTime(viagem.data_fim)}</td>
                  </tr>
                </tbody>
              </Table>
            </>
          )}
        </Card.Body>
      </Card>

      {/* Timeline de Logs */}
      <Card className="shadow-sm w-100">
        <Card.Header>Timeline de Logs</Card.Header>
        <Card.Body style={{ maxHeight: 400, overflowY: 'auto' }}>
          {logs.length === 0 ? (
            <p className="text-center">Nenhum log registrado.</p>
          ) : (
            logs.map((log, idx) => {
              const isUser = log.sender === 'user';
              const messageBgClass = isUser ? 'bg-success text-white' : 'bg-light text-dark border';
              const messageAlignClass = isUser ? 'justify-content-end' : 'justify-content-start';
              const outerRowAlignClass = isUser ? 'justify-content-end' : 'justify-content-start';

              return (
                <Row key={log.id} className={`mb-2 ${outerRowAlignClass}`}>
                  <Col xs="auto">
                    <div className={`d-flex flex-column ${messageAlignClass}`}>
                      <div className={`rounded-3 p-2 ${messageBgClass}`} style={{ maxWidth: '80%' }}>
                        <div className="small">{log.message}</div>
                      </div>
                      <div className={`d-flex align-items-center ${isUser ? 'justify-content-end' : 'justify-content-start'} mt-1`}>
                        <small className="text-muted me-2 small">{formatDateTime(log.createdAt)}</small>
                        {isUser && <span className="small" style={{ color: 'lightblue' }}>&#10004;&#10004;</span>}
                      </div>
                    </div>
                  </Col>
                </Row>
              );
            })
          )}
        </Card.Body>
      </Card>
    </Container>
  );
}
