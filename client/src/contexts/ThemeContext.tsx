import { useState, useEffect, useMemo, type ReactNode } from 'react';
import { ThemeContext, Theme, DEFAULT_THEME, THEMES } from './ThemeContextDefinition';

interface ThemeProviderProps {
  children: ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const [theme, setTheme] = useState<Theme>(() => {
    // Read from localStorage for persistence
    try {
      const storedTheme = window.localStorage.getItem('app-theme') as Theme;
      return THEMES.includes(storedTheme) ? storedTheme : DEFAULT_THEME;
    } catch (error) {
      console.error("Could not read theme from localStorage", error);
      return DEFAULT_THEME;
    }
  });

  useEffect(() => {
    // Apply the data-theme attribute to the root element
    document.documentElement.setAttribute('data-theme', theme);
    
    // Persist the theme choice
    try {
      window.localStorage.setItem('app-theme', theme);
    } catch (error) {
      console.error("Could not save theme to localStorage", error);
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => {
      const currentIndex = THEMES.indexOf(prevTheme);
      const nextIndex = (currentIndex + 1) % THEMES.length;
      return THEMES[nextIndex];
    });
  };

  const value = useMemo(
    () => ({
      theme,
      setTheme,
      toggleTheme,
      availableThemes: THEMES,
    }),
    [theme]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

