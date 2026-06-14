import { useState } from 'react'

export default function SourceList({ sources }) {
  const [showDetails, setShowDetails] = useState(false)

  if (!sources?.length) {
    return <p className="no-sources">No relevant source found.</p>
  }

  const preview = sources.slice(0, 3)
  const hasMore = sources.length > preview.length

  return (
    <div className="source-list">
      <div className="source-summary">
        {preview.map((source, index) => (
          <div key={`${source.source}-${index}`} className="source-line">
            <span className="source-file">{source.source}</span>
            <span className="source-meta-inline">
              {source.page != null ? `page ${source.page}` : 'page n/a'}
              {source.figure_number ? ` - ${source.figure_number}` : ''}
            </span>
          </div>
        ))}
      </div>

      {(hasMore || sources.some((source) => source.chunk_id || source.content_type)) ? (
        <button type="button" className="source-toggle" onClick={() => setShowDetails((value) => !value)}>
          {showDetails ? 'Hide source details' : 'Show source details'}
        </button>
      ) : null}

      {showDetails ? (
        <div className="source-details">
          {sources.map((source, index) => (
            <div key={`${source.source}-${index}`} className="source-detail-row">
              <span>{source.source}</span>
              <span>{source.page != null ? `page ${source.page}` : 'page n/a'}</span>
              <span>{source.figure_number || 'figure n/a'}</span>
              <span>{source.content_type}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}
