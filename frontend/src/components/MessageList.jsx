import MessageBubble from './MessageBubble'

export default function MessageList({ messages, error, isLoading }) {
  return (
    <section className="message-panel">
      <div className="message-list">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {isLoading ? (
          <div className="typing-row" aria-label="Assistant is typing">
            <span />
            <span />
            <span />
          </div>
        ) : null}
      </div>

      {error ? <div className="inline-error">{error}</div> : null}
    </section>
  )
}
