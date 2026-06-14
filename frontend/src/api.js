const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options)
  const contentType = response.headers.get('content-type') || ''
  const payload = contentType.includes('application/json')
    ? await response.json()
    : await response.text()

  if (!response.ok) {
    const message =
      typeof payload === 'object' && payload
        ? payload.detail || payload.message || 'Request failed'
        : 'Request failed'
    throw new Error(message)
  }

  return payload
}

export async function getHealth() {
  return request('/health')
}

export async function getDocuments() {
  return request('/documents')
}

export async function sendChat(question, sessionId) {
  return request('/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question,
      session_id: sessionId,
    }),
  })
}

export async function uploadFiles(files) {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))

  return request('/upload', {
    method: 'POST',
    body: formData,
  })
}

export async function rebuildIndex() {
  return request('/ingest', {
    method: 'POST',
  })
}

export { API_BASE_URL }
