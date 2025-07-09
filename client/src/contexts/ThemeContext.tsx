import React, { createContext, useContext, useState, useEffect, useMemo } from 'react';

// Define the theme names for TypeScript safety
export type Theme = 'celestial-indigo' | 'veridian-twilight' | 'solaris-flare' | 'hunters-vision' | 'nebula';
const THEMES: Theme[] = ['celestial-indigo', 'veridian-twilight', 'solaris-flare', 'hunters-vision', 'nebula'];

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  availableThemes: Theme[];
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// Define Celestial Indigo as the new default theme
const DEFAULT_THEME: Theme = 'celestial-indigo';

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
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
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};