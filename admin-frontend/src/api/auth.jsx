import axios from 'axios';
const API = process.env.REACT_APP_API_URL || 'http://localhost:3000/api';

export async function login(username, password) {
  const res = await axios.post(`${API}/auth/login`, { username, password });
  return res.data.access_token;
}
