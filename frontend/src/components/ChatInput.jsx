export default function ChatInput({ input, isLoading, onInputChange, onSend }) {
  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      onSend()
    }
  }

  return (
    <form
      className="chat-input-shell"
      onSubmit={(event) => {
        event.preventDefault()
        onSend()
      }}
    >
      <textarea
        value={input}
        onChange={(event) => onInputChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask anything about your documents..."
        rows={1}
      />

      <button type="submit" className="send-button" disabled={isLoading || !input.trim()}>
        {isLoading ? 'Sending...' : 'Send'}
      </button>
    </form>
  )
}
