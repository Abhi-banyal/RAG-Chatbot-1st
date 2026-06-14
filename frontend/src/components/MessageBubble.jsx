import SourceList from './SourceList'

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  return (
    <article className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-shell">
        <div className="message-label">{isUser ? 'You' : 'Assistant'}</div>
        <div className={`message-bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>
          <p className="message-text">{message.content}</p>

          {!isUser ? (
            <div className="assistant-extras">
              {message.needs_clarification ? (
                <div className="clarification-note">
                  Did you mean {message.suggested_question ? `"${message.suggested_question}"` : 'to ask a different question'}?
                </div>
              ) : null}

              <SourceList sources={message.sources || []} />
            </div>
          ) : null}
        </div>
      </div>
    </article>
  )
}
