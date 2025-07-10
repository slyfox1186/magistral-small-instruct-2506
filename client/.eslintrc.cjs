// Modern ESLint configuration for TypeScript + React + Vite (2025 best practices)
module.exports = {
  root: true,
  env: {
    browser: true,
    es2022: true,
    node: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended-type-checked',
    'plugin:@typescript-eslint/stylistic-type-checked',
    'plugin:react/recommended',
    'plugin:react/jsx-runtime',
    'plugin:react-hooks/recommended',
    'plugin:jsx-a11y/recommended',
    'plugin:import/recommended',
    'plugin:import/typescript',
    'plugin:prettier/recommended', // Must be last
  ],
  ignorePatterns: [
    'dist',
    '.eslintrc.cjs',
    'vite.config.ts',
    'node_modules',
    '*.config.js',
    '*.config.ts',
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    project: ['./tsconfig.json', './tsconfig.node.json'],
    tsconfigRootDir: '.',
    ecmaFeatures: {
      jsx: true,
    },
  },
  plugins: [
    'react-refresh',
    '@typescript-eslint',
    'react',
    'jsx-a11y',
    'import',
    'unicorn',
    'promise',
  ],
  settings: {
    react: {
      version: 'detect',
    },
    'import/resolver': {
      typescript: {
        project: './tsconfig.json',
      },
    },
  },
  rules: {
    // Disable no-undef globally for TypeScript projects - TypeScript handles this better
    'no-undef': 'off',
    
    // React Refresh
    'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],

    // TypeScript specific rules
    '@typescript-eslint/no-unused-vars': [
      'error',
      {
        argsIgnorePattern: '^_',
        varsIgnorePattern: '^_',
        caughtErrorsIgnorePattern: '^_',
      },
    ],
    '@typescript-eslint/consistent-type-imports': [
      'error',
      {
        prefer: 'type-imports',
        fixStyle: 'separate-type-imports',
      },
    ],
    '@typescript-eslint/explicit-function-return-type': 'off',
    '@typescript-eslint/explicit-module-boundary-types': 'off',
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/no-floating-promises': 'error',
    '@typescript-eslint/no-misused-promises': 'error',
    '@typescript-eslint/await-thenable': 'error',
    '@typescript-eslint/no-unnecessary-type-assertion': 'error',
    '@typescript-eslint/prefer-nullish-coalescing': 'error',
    '@typescript-eslint/prefer-optional-chain': 'error',

    // React rules
    'react/prop-types': 'off', // TypeScript handles this
    'react/jsx-uses-react': 'off',
    'react/react-in-jsx-scope': 'off',
    'react/jsx-no-target-blank': 'error',
    'react/jsx-curly-brace-presence': ['error', { props: 'never', children: 'never' }],

    // Import rules
    'import/order': [
      'error',
      {
        groups: ['builtin', 'external', 'internal', 'parent', 'sibling', 'index', 'object', 'type'],
        'newlines-between': 'always',
        pathGroups: [
          {
            pattern: '@/**',
            group: 'internal',
            position: 'after',
          },
        ],
        pathGroupsExcludedImportTypes: ['type'],
        alphabetize: {
          order: 'asc',
          caseInsensitive: true,
        },
      },
    ],
    'import/no-duplicates': 'error',
    'import/no-unresolved': 'error',
    'import/no-cycle': 'error',

    // Promise handling
    'promise/always-return': 'error',
    'promise/catch-or-return': 'error',
    'promise/no-nesting': 'warn',

    // Best practices
    'no-console': ['warn', { allow: ['warn', 'error'] }],
    'no-debugger': 'error',
    'no-alert': 'error',
    'no-var': 'error',
    'prefer-const': 'error',
    'prefer-template': 'error',
    'no-nested-ternary': 'error',
    'no-unneeded-ternary': 'error',

    // Unicorn rules for modern JS
    'unicorn/better-regex': 'error',
    'unicorn/catch-error-name': 'error',
    'unicorn/no-array-for-each': 'error',
    'unicorn/no-useless-undefined': 'error',
    'unicorn/prefer-array-find': 'error',
    'unicorn/prefer-array-some': 'error',
    'unicorn/prefer-includes': 'error',
    'unicorn/prefer-string-starts-ends-with': 'error',
    'unicorn/prefer-type-error': 'error',

    // Accessibility
    'jsx-a11y/no-autofocus': 'warn',
    'jsx-a11y/anchor-is-valid': 'error',
  },
};
