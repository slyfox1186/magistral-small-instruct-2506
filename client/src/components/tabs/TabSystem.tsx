import React, { useState, useRef, useEffect, useCallback } from 'react';
import { CSSTransition, TransitionGroup } from 'react-transition-group';

// Tab system types
export interface Tab {
  id: string;
  label: string;
  icon?: React.ReactNode;
  content: React.ReactNode;
  badge?: number | string;
  disabled?: boolean;
  closeable?: boolean;
  pinned?: boolean;
}

export interface TabSystemProps {
  tabs: Tab[];
  activeTabId: string;
  onTabChange: (tabId: string) => void;
  onTabClose?: (tabId: string) => void;
  onTabReorder?: (fromIndex: number, toIndex: number) => void;
  className?: string;
  variant?: 'primary' | 'secondary' | 'minimal';
  position?: 'top' | 'bottom' | 'left' | 'right';
  scrollable?: boolean;
  animated?: boolean;
  keyboardNavigation?: boolean;
  persistState?: boolean;
  maxTabs?: number;
}

const TabSystem: React.FC<TabSystemProps> = ({
  tabs,
  activeTabId,
  onTabChange,
  onTabClose,
  onTabReorder,
  className = '',
  variant = 'primary',
  position = 'top',
  scrollable = true,
  animated = true,
  keyboardNavigation = true,
  persistState = true,
  maxTabs: _maxTabs
}) => {
  const [draggedTab, setDraggedTab] = useState<string | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number>(-1);
  const tabListRef = useRef<HTMLDivElement>(null);
  const [scrollPosition, setScrollPosition] = useState(0);
  const [focusedTabIndex, setFocusedTabIndex] = useState(-1);

  // Persist tab state to localStorage
  useEffect(() => {
    if (persistState) {
      localStorage.setItem('tabSystem_activeTab', activeTabId);
    }
  }, [activeTabId, persistState]);

  // Load persisted tab state
  useEffect(() => {
    if (persistState) {
      const savedActiveTab = localStorage.getItem('tabSystem_activeTab');
      if (savedActiveTab && tabs.some(tab => tab.id === savedActiveTab)) {
        onTabChange(savedActiveTab);
      }
    }
  }, [persistState, tabs, onTabChange]);

  // Keyboard navigation
  useEffect(() => {
    if (!keyboardNavigation) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      const tabsCount = tabs.length;
      const currentIndex = tabs.findIndex(tab => tab.id === activeTabId);

      switch (event.key) {
        case 'ArrowLeft':
        case 'ArrowUp':
          event.preventDefault();
          const prevIndex = currentIndex > 0 ? currentIndex - 1 : tabsCount - 1;
          if (!tabs[prevIndex].disabled) {
            onTabChange(tabs[prevIndex].id);
            setFocusedTabIndex(prevIndex);
          }
          break;
        case 'ArrowRight':
        case 'ArrowDown':
          event.preventDefault();
          const nextIndex = currentIndex < tabsCount - 1 ? currentIndex + 1 : 0;
          if (!tabs[nextIndex].disabled) {
            onTabChange(tabs[nextIndex].id);
            setFocusedTabIndex(nextIndex);
          }
          break;
        case 'Home':
          event.preventDefault();
          const firstTab = tabs.find(tab => !tab.disabled);
          if (firstTab) {
            onTabChange(firstTab.id);
            setFocusedTabIndex(0);
          }
          break;
        case 'End':
          event.preventDefault();
          const lastTab = tabs.slice().reverse().find(tab => !tab.disabled);
          if (lastTab) {
            onTabChange(lastTab.id);
            setFocusedTabIndex(tabsCount - 1);
          }
          break;
        case 'Delete':
        case 'Backspace':
          if (onTabClose && tabs[currentIndex]?.closeable) {
            event.preventDefault();
            onTabClose(activeTabId);
          }
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [keyboardNavigation, tabs, activeTabId, onTabChange, onTabClose]);

  // Drag and drop handlers
  const handleDragStart = useCallback((e: React.DragEvent, tabId: string, _index: number) => {
    setDraggedTab(tabId);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', tabId);
    
    // Create drag image
    const dragImage = e.currentTarget.cloneNode(true) as HTMLElement;
    dragImage.style.opacity = '0.8';
    dragImage.style.transform = 'rotate(5deg)';
    document.body.appendChild(dragImage);
    e.dataTransfer.setDragImage(dragImage, 0, 0);
    setTimeout(() => document.body.removeChild(dragImage), 0);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent, index: number) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    setDragOverIndex(index);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent, dropIndex: number) => {
    e.preventDefault();
    const draggedTabId = e.dataTransfer.getData('text/plain');
    const draggedIndex = tabs.findIndex(tab => tab.id === draggedTabId);
    
    if (draggedIndex !== -1 && draggedIndex !== dropIndex && onTabReorder) {
      onTabReorder(draggedIndex, dropIndex);
    }
    
    setDraggedTab(null);
    setDragOverIndex(-1);
  }, [tabs, onTabReorder]);

  const handleDragEnd = useCallback(() => {
    setDraggedTab(null);
    setDragOverIndex(-1);
  }, []);

  // Scroll handlers for overflow tabs
  const scrollTabs = useCallback((direction: 'left' | 'right') => {
    if (!tabListRef.current) return;
    
    const scrollAmount = 200;
    const newPosition = direction === 'left' 
      ? Math.max(0, scrollPosition - scrollAmount)
      : scrollPosition + scrollAmount;
    
    tabListRef.current.scrollTo({
      left: newPosition,
      behavior: 'smooth'
    });
    setScrollPosition(newPosition);
  }, [scrollPosition]);

  const activeTab = tabs.find(tab => tab.id === activeTabId);

  return (
    <div className={`tab-system tab-system--${variant} tab-system--${position} ${className}`}>
      {/* Tab Navigation */}
      <div className="tab-nav">
        {scrollable && scrollPosition > 0 && (
          <button 
            className="tab-scroll-button tab-scroll-button--left"
            onClick={() => scrollTabs('left')}
            aria-label="Scroll tabs left"
          >
            ←
          </button>
        )}
        
        <div 
          ref={tabListRef}
          className="tab-list"
          role="tablist"
          aria-orientation={position === 'left' || position === 'right' ? 'vertical' : 'horizontal'}
        >
          {tabs.map((tab, index) => (
            <div
              key={tab.id}
              className={`
                tab-item 
                ${tab.id === activeTabId ? 'tab-item--active' : ''}
                ${tab.disabled ? 'tab-item--disabled' : ''}
                ${tab.pinned ? 'tab-item--pinned' : ''}
                ${draggedTab === tab.id ? 'tab-item--dragging' : ''}
                ${dragOverIndex === index ? 'tab-item--drag-over' : ''}
                ${focusedTabIndex === index ? 'tab-item--focused' : ''}
              `}
              role="tab"
              tabIndex={tab.disabled ? -1 : 0}
              aria-selected={tab.id === activeTabId}
              aria-disabled={tab.disabled}
              onClick={() => !tab.disabled && onTabChange(tab.id)}
              onFocus={() => setFocusedTabIndex(index)}
              onBlur={() => setFocusedTabIndex(-1)}
              draggable={!tab.disabled && !tab.pinned && !!onTabReorder}
              onDragStart={(e) => handleDragStart(e, tab.id, index)}
              onDragOver={(e) => handleDragOver(e, index)}
              onDrop={(e) => handleDrop(e, index)}
              onDragEnd={handleDragEnd}
            >
              {/* Tab Icon */}
              {tab.icon && (
                <span className="tab-icon" aria-hidden="true">
                  {tab.icon}
                </span>
              )}
              
              {/* Tab Label */}
              <span className="tab-label">{tab.label}</span>
              
              {/* Tab Badge */}
              {tab.badge && (
                <span className="tab-badge" aria-label={`${tab.badge} notifications`}>
                  {tab.badge}
                </span>
              )}
              
              {/* Close Button */}
              {tab.closeable && onTabClose && (
                <button
                  className="tab-close"
                  onClick={(e) => {
                    e.stopPropagation();
                    onTabClose(tab.id);
                  }}
                  aria-label={`Close ${tab.label} tab`}
                >
                  ×
                </button>
              )}
              
              {/* Active Indicator */}
              {tab.id === activeTabId && (
                <div className="tab-active-indicator" aria-hidden="true" />
              )}
            </div>
          ))}
        </div>
        
        {scrollable && (
          <button 
            className="tab-scroll-button tab-scroll-button--right"
            onClick={() => scrollTabs('right')}
            aria-label="Scroll tabs right"
          >
            →
          </button>
        )}
      </div>
      
      {/* Tab Content */}
      <div className="tab-content" role="tabpanel" aria-labelledby={`tab-${activeTabId}`}>
        {!activeTab ? (
          <div className="tab-panel">
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
              No content available for tab: {activeTabId}
            </div>
          </div>
        ) : animated ? (
          <TransitionGroup component={null}>
            <CSSTransition
              key={activeTab.id}
              timeout={300}
              classNames="tab-content-transition"
              appear
            >
              <div className="tab-panel">
                {activeTab.content}
              </div>
            </CSSTransition>
          </TransitionGroup>
        ) : (
          <div className="tab-panel">
            {activeTab.content}
          </div>
        )}
      </div>
    </div>
  );
};

export default TabSystem;