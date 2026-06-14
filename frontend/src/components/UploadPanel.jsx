import { useRef } from 'react'

export default function UploadPanel({ uploadStatus, rebuildStatus, onUpload, onRebuild }) {
  const inputRef = useRef(null)

  return (
    <section className="sidebar-section upload-section">
      <div className="section-title">Documents</div>

      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".pdf,.txt,.png,.jpg,.jpeg"
        onChange={(event) => onUpload(Array.from(event.target.files || []))}
        className="file-input"
      />

      <div className="sidebar-actions">
        <button type="button" className="sidebar-button secondary" onClick={() => inputRef.current?.click()}>
          Upload files
        </button>
        <button type="button" className="sidebar-button primary" onClick={onRebuild}>
          Rebuild index
        </button>
      </div>

      {uploadStatus ? <p className="sidebar-note">{uploadStatus}</p> : null}
      {rebuildStatus ? <p className="sidebar-note">{rebuildStatus}</p> : null}
    </section>
  )
}
