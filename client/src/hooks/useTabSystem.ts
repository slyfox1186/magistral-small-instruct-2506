import { useState, useCallback, useEffect, useRef } from 'react';

export interface TabState {
  activeTabId: string;
  tabs: string[];
  history: string[];
}

export interface UseTabSystemOptions {
  initialTabId?: string;
  persistKey?: string;
  maxHistory?: number;
  onTabChange?: (tabId: string) => void;
  onTabClose?: (tabId: string) => void;
}

export const useTabSystem = (options: UseTabSystemOptions = {}) => {
  const {
    initialTabId = '',
    persistKey,
    maxHistory = 10,
    onTabChange,
    onTabClose
  } = options;

  const [activeTabId, setActiveTabId] = useState<string>(initialTabId);
  const [tabHistory, setTabHistory] = useState<string[]>([]);
  const [tabOrder, setTabOrder] = useState<string[]>([]);
  const previousTabRef = useRef<string>('');

  // Load persisted state
  useEffect(() => {
    if (persistKey) {
      try {
        const saved = localStorage.getItem(`tabSystem_${persistKey}`);
        if (saved) {
          const state: TabState = JSON.parse(saved);
          setActiveTabId(state.activeTabId || initialTabId);
          setTabHistory(state.history || []);
          setTabOrder(state.tabs || []);
        }
      } catch (error) {
        console.warn('Failed to load tab state:', error);
      }
    }
  }, [persistKey, initialTabId]);

  // Persist state changes
  useEffect(() => {
    if (persistKey && activeTabId) {
      try {
        const state: TabState = {
          activeTabId,
          tabs: tabOrder,
          history: tabHistory
        };
        localStorage.setItem(`tabSystem_${persistKey}`, JSON.stringify(state));
      } catch (error) {
        console.warn('Failed to save tab state:', error);
      }
    }
  }, [persistKey, activeTabId, tabOrder, tabHistory]);

  // Update history when tab changes
  useEffect(() => {
    if (activeTabId && activeTabId !== previousTabRef.current) {
      setTabHistory(prev => {
        const newHistory = [activeTabId, ...prev.filter(id => id !== activeTabId)];
        return newHistory.slice(0, maxHistory);
      });
      previousTabRef.current = activeTabId;
    }
  }, [activeTabId, maxHistory]);

  // Change active tab
  const changeTab = useCallback((tabId: string) => {
    setActiveTabId(tabId);
    onTabChange?.(tabId);
  }, [onTabChange]);

  // Close a tab
  const closeTab = useCallback((tabId: string) => {
    // If closing active tab, switch to previous or next available tab
    if (tabId === activeTabId) {
      const currentIndex = tabOrder.indexOf(tabId);
      const remainingTabs = tabOrder.filter(id => id !== tabId);
      
      if (remainingTabs.length > 0) {
        // Try to use the most recent tab from history
        const nextTab = tabHistory.find(id => id !== tabId && remainingTabs.includes(id)) ||
                       remainingTabs[Math.min(currentIndex, remainingTabs.length - 1)];
        setActiveTabId(nextTab);
      } else {
        setActiveTabId('');
      }
    }

    // Update tab order
    setTabOrder(prev => prev.filter(id => id !== tabId));
    
    // Clean up history
    setTabHistory(prev => prev.filter(id => id !== tabId));
    
    onTabClose?.(tabId);
  }, [activeTabId, tabOrder, tabHistory, onTabClose]);

  // Add a new tab
  const addTab = useCallback((tabId: string, position?: number) => {
    setTabOrder(prev => {
      if (prev.includes(tabId)) {
        return prev; // Tab already exists
      }
      
      if (position !== undefined && position >= 0 && position <= prev.length) {
        const newOrder = [...prev];
        newOrder.splice(position, 0, tabId);
        return newOrder;
      }
      
      return [...prev, tabId];
    });
  }, []);

  // Reorder tabs
  const reorderTabs = useCallback((fromIndex: number, toIndex: number) => {
    setTabOrder(prev => {
      const newOrder = [...prev];
      const [movedTab] = newOrder.splice(fromIndex, 1);
      newOrder.splice(toIndex, 0, movedTab);
      return newOrder;
    });
  }, []);

  // Get previous tab
  const getPreviousTab = useCallback(() => {
    return tabHistory[1] || tabOrder.find(id => id !== activeTabId) || '';
  }, [tabHistory, tabOrder, activeTabId]);

  // Navigate to previous tab
  const goToPreviousTab = useCallback(() => {
    const previousTab = getPreviousTab();
    if (previousTab) {
      changeTab(previousTab);
    }
  }, [getPreviousTab, changeTab]);

  // Navigate to next tab
  const goToNextTab = useCallback(() => {
    const currentIndex = tabOrder.indexOf(activeTabId);
    const nextIndex = (currentIndex + 1) % tabOrder.length;
    const nextTab = tabOrder[nextIndex];
    if (nextTab) {
      changeTab(nextTab);
    }
  }, [tabOrder, activeTabId, changeTab]);

  // Check if tab exists
  const hasTab = useCallback((tabId: string) => {
    return tabOrder.includes(tabId);
  }, [tabOrder]);

  // Get tab position
  const getTabPosition = useCallback((tabId: string) => {
    return tabOrder.indexOf(tabId);
  }, [tabOrder]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeydown = (event: KeyboardEvent) => {
      // Only handle if not typing in an input
      if (event.target instanceof HTMLInputElement || 
          event.target instanceof HTMLTextAreaElement ||
          event.target instanceof HTMLSelectElement) {
        return;
      }

      // Ctrl/Cmd + Tab - Next tab
      if ((event.ctrlKey || event.metaKey) && event.key === 'Tab' && !event.shiftKey) {
        event.preventDefault();
        goToNextTab();
      }
      
      // Ctrl/Cmd + Shift + Tab - Previous tab
      if ((event.ctrlKey || event.metaKey) && event.key === 'Tab' && event.shiftKey) {
        event.preventDefault();
        goToPreviousTab();
      }

      // Ctrl/Cmd + W - Close current tab
      if ((event.ctrlKey || event.metaKey) && event.key === 'w') {
        event.preventDefault();
        if (activeTabId) {
          closeTab(activeTabId);
        }
      }

      // Ctrl/Cmd + 1-9 - Switch to tab by number
      if ((event.ctrlKey || event.metaKey) && /^[1-9]$/.test(event.key)) {
        event.preventDefault();
        const tabIndex = parseInt(event.key) - 1;
        const targetTab = tabOrder[tabIndex];
        if (targetTab) {
          changeTab(targetTab);
        }
      }
    };

    document.addEventListener('keydown', handleKeydown);
    return () => document.removeEventListener('keydown', handleKeydown);
  }, [goToNextTab, goToPreviousTab, closeTab, activeTabId, tabOrder, changeTab]);

  return {
    // State
    activeTabId,
    tabOrder,
    tabHistory,
    
    // Actions
    changeTab,
    closeTab,
    addTab,
    reorderTabs,
    goToPreviousTab,
    goToNextTab,
    
    // Utilities
    hasTab,
    getTabPosition,
    getPreviousTab,
    
    // Computed
    tabCount: tabOrder.length,
    canGoBack: tabHistory.length > 1,
    canGoForward: tabOrder.length > 1
  };
};