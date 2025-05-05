import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  AiFillHome,
  AiOutlineUnorderedList,
  AiOutlineUsergroupAdd,
  AiOutlineUser,
} from 'react-icons/ai';

const linkStyle = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  textDecoration: 'none',
  color: 'var(--text-soft)',
  padding: 'var(--spacing-xs) 0',
};
const activeStyle = { color: 'var(--color-primary)' };
const labelStyle  = { fontSize: '12px', marginTop: 'var(--spacing-xs)' };

export default function NavBar() {
  return (
    <nav
      style={{
        width: '100%',
        borderTop: '1px solid var(--border-color)',
        background: 'var(--color-secondary)',
        padding: 'var(--spacing-xs) 0',
      }}
    >
      <div className="container">
        <ul
          style={{
            display: 'flex',
            justifyContent: 'space-around',
            margin: 0,
            padding: 0,
            listStyle: 'none',
          }}
        >
          <li>
            <NavLink to="/" end style={linkStyle} activeStyle={activeStyle}>
              <AiFillHome size={24} />
              <span style={labelStyle}>Início</span>
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/sessions"
              style={linkStyle}
              activeStyle={activeStyle}
            >
              <AiOutlineUnorderedList size={24} />
              <span style={labelStyle}>Sessões</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/leads" style={linkStyle} activeStyle={activeStyle}>
              <AiOutlineUsergroupAdd size={24} />
              <span style={labelStyle}>Leads</span>
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/admin-users"
              style={linkStyle}
              activeStyle={activeStyle}
            >
              <AiOutlineUser size={24} />
              <span style={labelStyle}>Admins</span>
            </NavLink>
          </li>
        </ul>
      </div>
    </nav>
  );
}
