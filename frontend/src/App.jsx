import { useEffect, useMemo, useState } from 'react'
import { getDocuments, getHealth, sendChat, uploadFiles } from './api'
import ChatInput from './components/ChatInput'
import ChatLayout from './components/ChatLayout'
import Header from './components/Header'
import MessageList from './components/MessageList'
import Sidebar from './components/Sidebar'

const STORAGE_KEY = 'rag-chatbot-session-id'

function getSessionId() {
  const existing = window.localStorage.getItem(STORAGE_KEY)
  if (existing) {
    return existing
  }

  const generated = window.crypto?.randomUUID?.() || `session-${Date.now()}`
  window.localStorage.setItem(STORAGE_KEY, generated)
  return generated
}

function getMessagesKey(sessionId) {
  return `rag-chatbot-messages-${sessionId}`
}

function loadMessages(sessionId) {
  try {
    const raw = window.localStorage.getItem(getMessagesKey(sessionId))
    if (!raw) {
      return []
    }

    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function saveMessages(sessionId, messages) {
  window.localStorage.setItem(getMessagesKey(sessionId), JSON.stringify(messages))
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => getSessionId())
  const [messages, setMessages] = useState(() => loadMessages(getSessionId()))
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [health, setHealth] = useState(null)
  const [documents, setDocuments] = useState([])
  const [uploadStatus, setUploadStatus] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const assistantCount = useMemo(
    () => messages.filter((message) => message.role === 'assistant').length,
    [messages],
  )

  async function refreshDocuments() {
    const response = await getDocuments()
    setDocuments(response.documents || [])
  }

  useEffect(() => {
    let active = true

    async function bootstrap() {
      try {
        const [healthResponse, documentsResponse] = await Promise.all([getHealth(), getDocuments()])
        if (!active) {
          return
        }
        setHealth(healthResponse)
        setDocuments(documentsResponse.documents || [])
      } catch (err) {
        if (!active) {
          return
        }
        setError(err.message || 'Unable to connect to backend')
      }
    }

    bootstrap()

    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    saveMessages(sessionId, messages)
  }, [sessionId, messages])

  function startNewChat() {
    const nextSessionId = window.crypto?.randomUUID?.() || `session-${Date.now()}`
    window.localStorage.setItem(STORAGE_KEY, nextSessionId)
    setSessionId(nextSessionId)
    setMessages([])
    setInput('')
    setError('')
    setUploadStatus('')
    setSidebarOpen(false)
  }

  async function handleSend() {
    const question = input.trim()
    if (!question || isLoading) {
      return
    }

    setError('')
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: 'user',
        content: question,
        sources: [],
      },
    ])
    setInput('')
    setIsLoading(true)

    try {
      const response = await sendChat(question, sessionId)
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: response.answer,
          sources: response.sources || [],
          needs_clarification: response.needs_clarification,
          suggested_question: response.suggested_question,
        },
      ])
    } catch (err) {
      setError(err.message || 'Chat request failed')
    } finally {
      setIsLoading(false)
    }
  }

  async function handleUpload(files) {
    if (!files?.length) {
      return
    }

    setError('')
    setUploadStatus('Uploading files...')

    try {
      const response = await uploadFiles(files)
      const uploaded = response.uploaded_files?.length || 0
      const failed = response.failed_files?.length || 0
      setUploadStatus(
        failed > 0
          ? `Uploaded and indexed ${uploaded} file(s). ${failed} file(s) were skipped.`
          : `Uploaded and indexed ${uploaded} file(s) successfully.`,
      )
      await refreshDocuments()
    } catch (err) {
      setUploadStatus('')
      setError(err.message || 'Upload failed')
    }
  }

  return (
    <div className={`app-shell ${sidebarOpen ? 'sidebar-open' : ''}`}>
      <Sidebar
        appName="Document RAG"
        health={health}
        documents={documents}
        uploadStatus={uploadStatus}
        sessionId={sessionId}
        onUpload={handleUpload}
        onNewChat={startNewChat}
        onCloseMobile={() => setSidebarOpen(false)}
        isOpen={sidebarOpen}
      />

      <div className="chat-shell">
        <Header
          health={health}
          assistantCount={assistantCount}
          onMenuClick={() => setSidebarOpen(true)}
          onNewChat={startNewChat}
          title="Document RAG Assistant"
          subtitle="Ask questions from your uploaded documents"
        />

        <ChatLayout>
          {messages.length === 0 ? (
            <div className="empty-stage">
              <div className="empty-copy">
                <h1>Where should we begin?</h1>
                <p className="empty-subtitle">Ask anything about your documents...</p>
              </div>

              <div className="empty-composer">
                <ChatInput
                  input={input}
                  isLoading={isLoading}
                  onInputChange={setInput}
                  onSend={handleSend}
                />
              </div>
            </div>
          ) : (
            <>
              <MessageList messages={messages} error={error} isLoading={isLoading} />
              <div className="composer-shell">
                <ChatInput
                  input={input}
                  isLoading={isLoading}
                  onInputChange={setInput}
                  onSend={handleSend}
                />
              </div>
            </>
          )}
        </ChatLayout>
      </div>
    </div>
  )
}
