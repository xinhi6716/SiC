const { contextBridge } = require('electron');

const API_BASE_URL = 'http://127.0.0.1:8000';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    }
  });

  const contentType = response.headers.get('content-type') || '';
  const payload = contentType.includes('application/json')
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message = typeof payload === 'object' && payload !== null
      ? payload.detail || JSON.stringify(payload)
      : payload;
    throw new Error(message || `Request failed with status ${response.status}`);
  }

  return payload;
}

contextBridge.exposeInMainWorld('api', {
  baseUrl: API_BASE_URL,

  fetchData: (path, options = {}) => request(path, options),

  getMaterialConfig: (materialName) => request(`/api/config/${encodeURIComponent(materialName)}`),

  predictMaterial: (materialName, features) => request(
    `/api/predict/${encodeURIComponent(materialName)}`,
    {
      method: 'POST',
      body: JSON.stringify({ features })
    }
  ),

  getRecipes: (materialName) => request(`/api/recipes/${encodeURIComponent(materialName)}`),

  healthCheck: () => request('/api/health')
});
