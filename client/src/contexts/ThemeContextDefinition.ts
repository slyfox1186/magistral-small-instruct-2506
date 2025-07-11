import { createContext } from 'react';

// Define the theme names for TypeScript safety
export type Theme = 'celestial-indigo' | 'veridian-twilight' | 'solaris-flare' | 'hunters-vision' | 'nebula' | 'crimson-ember' | 'cyberpunk-neon' | 'obsidian-slate';

export interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  availableThemes: Theme[];
}

export const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const THEMES: Theme[] = ['celestial-indigo', 'veridian-twilight', 'solaris-flare', 'hunters-vision', 'nebula', 'crimson-ember', 'cyberpunk-neon', 'obsidian-slate'];

// Define Celestial Indigo as the new default theme
export const DEFAULT_THEME: Theme = 'celestial-indigo';