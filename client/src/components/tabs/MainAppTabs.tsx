import React, { useState, useCallback } from 'react';
import TabSystem, { Tab } from './TabSystem';
import './TabSystem.css';

// Icons for different tabs
const ChatIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </svg>
);

const DatabaseIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <ellipse cx="12" cy="5" rx="9" ry="3" />
    <path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5" />
    <path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3" />
  </svg>
);

const AnalyticsIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M3 3v18h18" />
    <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3" />
  </svg>
);

const SystemIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
    <line x1="8" y1="21" x2="16" y2="21" />
    <line x1="12" y1="17" x2="12" y2="21" />
  </svg>
);

interface MainAppTabsProps {
  currentMode: 'chat' | 'crud';
  onModeChange: (mode: 'chat' | 'crud') => void;
  chatContent: React.ReactNode;
  crudContent: React.ReactNode;
  className?: string;
}

const MainAppTabs: React.FC<MainAppTabsProps> = ({
  currentMode,
  onModeChange,
  chatContent,
  crudContent,
  className = ''
}) => {
  const [activeTabId, setActiveTabId] = useState(currentMode === 'chat' ? 'chat' : 'data');
  const [messageCount] = useState(0);
  const [systemAlerts] = useState(0);

  // Handle tab changes
  const handleTabChange = useCallback((tabId: string) => {
    console.log('MainAppTabs: Switching to tab:', tabId);
    setActiveTabId(tabId);
    
    // Map tab IDs to interface modes
    switch (tabId) {
      case 'chat':
        console.log('MainAppTabs: Setting mode to chat');
        onModeChange('chat');
        break;
      case 'data':
        console.log('MainAppTabs: Setting mode to crud');
        onModeChange('crud');
        break;
      case 'analytics':
        console.log('MainAppTabs: Analytics tab selected - standalone content');
        // For now, analytics is a standalone tab - no mode change needed
        // The content is self-contained in the tab
        break;
      case 'system':
        console.log('MainAppTabs: System tab selected - standalone content');
        // For now, system is a standalone tab - no mode change needed  
        // The content is self-contained in the tab
        break;
    }
  }, [onModeChange]);

  // Update active tab when mode changes externally (only for chat/data tabs)
  React.useEffect(() => {
    const newTabId = currentMode === 'chat' ? 'chat' : 'data';
    // Only sync if current tab is chat or data, leave analytics/system alone
    if (newTabId !== activeTabId && (activeTabId === 'chat' || activeTabId === 'data')) {
      console.log('MainAppTabs: External mode change, updating tab from', activeTabId, 'to', newTabId);
      setActiveTabId(newTabId);
    }
  }, [currentMode, activeTabId]);

  // Mock analytics content
  const analyticsContent = (
    <div className="analytics-content" style={{ height: '100%', overflow: 'auto' }}>
      <div style={{ 
        padding: '2rem', 
        textAlign: 'center', 
        color: 'var(--text-secondary, #cbd5e1)',
        background: 'var(--bg-secondary, #252730)',
        borderRadius: '8px',
        margin: '2rem',
        minHeight: '200px'
      }}>
        <AnalyticsIcon />
        <h3 style={{ margin: '1rem 0', color: 'var(--text-primary, #f8fafc)' }}>
          Analytics Dashboard
        </h3>
        <p>Conversation insights, usage patterns, and performance metrics will be displayed here.</p>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
          gap: '1rem',
          marginTop: '2rem'
        }}>
          <div style={{ 
            padding: '1rem', 
            background: 'var(--bg-tertiary, #2f3349)', 
            borderRadius: '6px',
            border: '1px solid var(--border-color, #475569)'
          }}>
            <strong style={{ color: 'var(--accent-primary, #fbbf24)' }}>Total Conversations</strong>
            <div style={{ fontSize: '1.5rem', marginTop: '0.5rem' }}>247</div>
          </div>
          <div style={{ 
            padding: '1rem', 
            background: 'var(--bg-tertiary, #2f3349)', 
            borderRadius: '6px',
            border: '1px solid var(--border-color, #475569)'
          }}>
            <strong style={{ color: 'var(--accent-success, #22c55e)' }}>Avg Response Time</strong>
            <div style={{ fontSize: '1.5rem', marginTop: '0.5rem' }}>1.2s</div>
          </div>
          <div style={{ 
            padding: '1rem', 
            background: 'var(--bg-tertiary, #2f3349)', 
            borderRadius: '6px',
            border: '1px solid var(--border-color, #475569)'
          }}>
            <strong style={{ color: 'var(--accent-info, #7C9885)' }}>Memory Usage</strong>
            <div style={{ fontSize: '1.5rem', marginTop: '0.5rem' }}>78%</div>
          </div>
        </div>
      </div>
    </div>
  );

  // Mock system content
  const systemContent = (
    <div className="system-content" style={{ height: '100%', overflow: 'auto' }}>
      <div style={{ 
        padding: '2rem', 
        textAlign: 'center', 
        color: 'var(--text-secondary, #cbd5e1)',
        background: 'var(--bg-secondary, #252730)',
        borderRadius: '8px',
        margin: '2rem',
        minHeight: '200px' // Add minimum height for debugging
      }}>
        <SystemIcon />
        <h3 style={{ margin: '1rem 0', color: 'var(--text-primary, #f8fafc)' }}>
          System Health
        </h3>
        <p>System monitoring, health checks, and configuration options.</p>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
          gap: '1rem',
          marginTop: '2rem'
        }}>
          <div style={{ 
            padding: '1rem', 
            background: 'var(--bg-tertiary, #2f3349)', 
            borderRadius: '6px',
            border: '1px solid var(--border-color, #475569)'
          }}>
            <strong style={{ color: 'var(--accent-success, #22c55e)' }}>Backend Status</strong>
            <div style={{ 
              fontSize: '1.2rem', 
              marginTop: '0.5rem',
              color: 'var(--accent-success, #22c55e)'
            }}>
              🟢 Online
            </div>
          </div>
          <div style={{ 
            padding: '1rem', 
            background: 'var(--bg-tertiary, #2f3349)', 
            borderRadius: '6px',
            border: '1px solid var(--border-color, #475569)'
          }}>
            <strong style={{ color: 'var(--accent-success, #22c55e)' }}>Database</strong>
            <div style={{ 
              fontSize: '1.2rem', 
              marginTop: '0.5rem',
              color: 'var(--accent-success, #22c55e)'
            }}>
              🟢 Connected
            </div>
          </div>
          <div style={{ 
            padding: '1rem', 
            background: 'var(--bg-tertiary, #2f3349)', 
            borderRadius: '6px',
            border: '1px solid var(--border-color, #475569)'
          }}>
            <strong style={{ color: 'var(--accent-warning, #fbbf24)' }}>Memory Usage</strong>
            <div style={{ 
              fontSize: '1.2rem', 
              marginTop: '0.5rem',
              color: 'var(--accent-warning, #fbbf24)'
            }}>
              🟡 78%
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const tabs: Tab[] = [
    {
      id: 'chat',
      label: 'Chat',
      icon: <ChatIcon />,
      content: chatContent,
      badge: messageCount > 0 ? messageCount : undefined
    },
    {
      id: 'data',
      label: 'Data Management',
      icon: <DatabaseIcon />,
      content: crudContent
    },
    {
      id: 'analytics',
      label: 'Analytics',
      icon: <AnalyticsIcon />,
      content: analyticsContent
    },
    {
      id: 'system',
      label: 'System',
      icon: <SystemIcon />,
      content: systemContent,
      badge: systemAlerts > 0 ? systemAlerts : undefined
    }
  ];

  return (
    <TabSystem
      tabs={tabs}
      activeTabId={activeTabId}
      onTabChange={handleTabChange}
      className={`main-app-tabs ${className}`}
      variant="primary"
      position="top"
      scrollable={false}
      animated={true}
      keyboardNavigation={true}
      persistState={true}
    />
  );
};

export default MainAppTabs;