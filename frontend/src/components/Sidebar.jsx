import UploadPanel from './UploadPanel'

export default function Sidebar({
  appName,
  health,
  documents,
  uploadStatus,
  sessionId,
  onUpload,
  onNewChat,
  onCloseMobile,
  isOpen,
}) {
  const isOnline = Boolean(health)

  return (
    <>
      <button
        type="button"
        className={`mobile-overlay ${isOpen ? 'show' : ''}`}
        onClick={onCloseMobile}
        aria-label="Close sidebar"
      />

      <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-top">
          <div className="sidebar-brand">
            <span className="brand-mark" />
            <div>
              <p className="sidebar-name">{appName}</p>
              <p className={`backend-pill ${isOnline ? 'online' : 'offline'}`}>
                Backend: {isOnline ? 'Online' : 'Offline'}
              </p>
            </div>
          </div>

          <button type="button" className="new-chat-button" onClick={onNewChat}>
            + New Chat
          </button>
        </div>

        <UploadPanel
          documents={documents}
          uploadStatus={uploadStatus}
          onUpload={onUpload}
        />

        <div className="sidebar-section">
          <div className="section-title">Indexed files</div>
          <div className="sidebar-doc-list">
            {documents.length ? (
              documents.map((doc) => (
                <div key={`${doc.location}-${doc.name}`} className="sidebar-doc-item">
                  <span className="doc-name">{doc.name}</span>
                  <span className="doc-type">
                    {doc.location} · {doc.file_type}
                  </span>
                </div>
              ))
            ) : (
              <div className="empty-sidebar-state">No documents indexed yet.</div>
            )}
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="sidebar-session-label">Session</div>
          <div className="sidebar-session-id">{sessionId.slice(0, 12)}</div>
        </div>
      </aside>
    </>
  )
}
