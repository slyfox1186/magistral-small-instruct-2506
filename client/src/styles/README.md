# Styles Directory

This directory contains all CSS stylesheets for the Neural Consciousness Chat System frontend.

## File Structure

- **`index.css`** - Global application styles and CSS variables
- **`App.css`** - Main application layout and chat interface styles
- **`MarkdownIt.css`** - Markdown rendering styles including tables, code blocks, and content formatting

## Import Path

All CSS files are now imported from the `/styles/` directory:

```tsx
// From main.tsx
import './styles/index.css';

// From App.tsx  
import './styles/App.css';

// From components
import '../styles/MarkdownIt.css';
```

## Key Features

### App.css
- Chat bubble styling with proper padding
- Theme-aware color system using CSS variables
- Responsive layout and navigation
- Dark theme optimized interface

### MarkdownIt.css
- Comprehensive markdown element styling
- Table formatting with proper alignment (left-aligned attribute columns)
- Code syntax highlighting support
- Responsive table containers with horizontal scrolling

### index.css
- Global CSS reset and base styles
- CSS custom properties for theming
- Font loading and typography settings

## Notes

- All table attribute columns (first column) are left-aligned for better readability
- CSS follows a theme-aware design system using CSS custom properties
- Styles are optimized for dark theme usage