// src/pages/Dashboard.jsx
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Container, Row, Col, Card, Spinner } from 'react-bootstrap';
import { Bar } from 'react-chartjs-2';
import Chart from 'chart.js/auto';

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    axios.get('/api/metrics')
      .then(res => setMetrics(res.data))
      .catch(err => console.error(err));
  }, []);

  if (!metrics) {
    return (
      <Container className="text-center py-5">
        <Spinner animation="border" />
      </Container>
    );
  }

  const {
    totalSessions,
    totalMessages,
    totalDestinations,
    conversionRate,
    avgDurationMinutes,
    dailyVolume
  } = metrics;

  const labels = dailyVolume.map(d => d.date.slice(5)); // “MM-DD”
  const data   = dailyVolume.map(d => d.count);

  return (
    <Container className="py-4">
      <h2>Dashboard de Métricas</h2>

      {/* Linha 1: quatro métricas */}
      <Row className="gy-4 mb-4">
        <Col md={3}>
          <Card className="text-center">
            <Card.Body>
              <Card.Title>Total Sessões</Card.Title>
              <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
                {totalSessions}
              </div>
            </Card.Body>
          </Card>
        </Col>

        <Col md={3}>
          <Card className="text-center">
            <Card.Body>
              <Card.Title>Total Mensagens</Card.Title>
              <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
                {totalMessages}
              </div>
            </Card.Body>
          </Card>
        </Col>

        <Col md={3}>
          <Card className="text-center">
            <Card.Body>
              <Card.Title>Destinos Únicos</Card.Title>
              <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
                {totalDestinations}
              </div>
            </Card.Body>
          </Card>
        </Col>

        <Col md={3}>
          <Card className="text-center">
            <Card.Body>
              <Card.Title>Taxa de Conversão</Card.Title>
              <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
                {conversionRate.toFixed(1)}%
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Linha 2: tempo médio e gráfico de volume */}
      <Row className="gy-4">
        <Col md={4}>
          <Card className="text-center">
            <Card.Body>
              <Card.Title>Tempo Médio (min)</Card.Title>
              <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>
                {avgDurationMinutes.toFixed(1)}
              </div>
            </Card.Body>
          </Card>
        </Col>

        <Col md={8}>
          <Card>
            <Card.Header>Volume Diário (últimos 7 dias)</Card.Header>
            <Card.Body style={{ height: '250px' }}>
              <Bar
                data={{
                  labels,
                  datasets: [{
                    label: 'Sessões',
                    data,
                    backgroundColor: 'rgba(31, 58, 52, 0.7)'
                  }]
                }}
                options={{
                  maintainAspectRatio: false,
                  scales: { y: { beginAtZero: true } }
                }}
              />
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}
