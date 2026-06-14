export default function Header({
  health,
  assistantCount,
  onMenuClick,
  onNewChat,
  title,
  subtitle,
}) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <button type="button" className="icon-button mobile-only" onClick={onMenuClick} aria-label="Open sidebar">
          ☰
        </button>

        <div>
          <p className="topbar-title">{title}</p>
          <p className="topbar-subtitle">{subtitle}</p>
        </div>
      </div>

      <div className="topbar-right">
        <button type="button" className="ghost-button desktop-only" onClick={onNewChat}>
          New chat
        </button>
        <div className={`status-dot ${health ? 'online' : 'offline'}`} title={health ? 'Backend online' : 'Backend offline'} />
        <span className="status-text">{health ? 'Online' : 'Offline'}</span>
        <span className="message-count">{assistantCount} replies</span>
      </div>
    </header>
  )
}
