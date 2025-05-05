import React, { useState } from 'react';
import logo from '../assets/logo.png';
import { AiOutlineSearch } from 'react-icons/ai';

export default function TopBar() {
  const [query, setQuery] = useState('');

  return (
    // Header full-width com background
    <header
      style={{
        width: '100%',
        backgroundColor: 'var(--color-primary)',
        padding: 'var(--spacing-sm) 0',
      }}
    >
      {/* Conteúdo centralizado */}
      <div className="container"
           style={{
             display: 'flex',
             alignItems: 'center',
             justifyContent: 'space-between',
           }}
      >
        {/* Logo */}
        <img
          src={logo}
          alt="Logo"
          style={{ height: '52px' }}
        />

        {/* Search */}
        <div
          style={{
            position: 'relative',
            flex: 1,
            maxWidth: '400px',
            marginLeft: 'var(--spacing-md)',
          }}
        >
          <AiOutlineSearch
            className="search-icon"
            size={20}
            style={{ color: 'var(--color-secondary)' }}
          />
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Buscar..."
            style={{
              width: '100%',
              padding: '8px var(--spacing-sm) 8px calc(1em + var(--spacing-xl))',
              borderRadius: 'var(--radius-pill)',
              border: '1px solid var(--border-color)',
              fontSize: 'var(--font-size-body)',
            }}
          />
        </div>
      </div>
    </header>
  );
}
