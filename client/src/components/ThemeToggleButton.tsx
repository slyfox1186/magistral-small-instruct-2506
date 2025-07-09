import React from 'react';
import { useTheme } from '../contexts/ThemeContext';

export const ThemeToggleButton: React.FC = () => {
  const { theme, toggleTheme, availableThemes } = useTheme();

  const getThemeDisplayName = (themeName: string) => {
    switch (themeName) {
      case 'celestial-indigo':
        return 'Forest Dusk';
      case 'veridian-twilight':
        return 'Veridian Twilight';
      case 'solaris-flare':
        return 'Solaris Flare';
      case 'hunters-vision':
        return 'Hunter\'s Vision';
      case 'nebula':
        return 'Celestial Indigo';
      case 'crimson-ember':
        return 'Crimson Ember';
      default:
        return themeName;
    }
  };

  const getNextThemeName = () => {
    const currentIndex = availableThemes.indexOf(theme);
    const nextIndex = (currentIndex + 1) % availableThemes.length;
    return getThemeDisplayName(availableThemes[nextIndex]);
  };

  return (
    <button 
      onClick={toggleTheme}
      className="theme-toggle-button"
      aria-label={`Switch to ${getNextThemeName()} theme`}
      title={`Currently: ${getThemeDisplayName(theme)}. Click to switch to ${getNextThemeName()}`}
    >
      🎨 {getThemeDisplayName(theme)}
    </button>
  );
};