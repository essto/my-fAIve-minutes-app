export default [
    {
        languageOptions: {
            parser: (await import('@typescript-eslint/parser')).default,
            parserOptions: {
                ecmaVersion: 'latest',
                sourceType: 'module',
            },
        },
        plugins: {
            '@typescript-eslint': (await import('@typescript-eslint/eslint-plugin')).default,
        },
        rules: {
            'no-console': 'error',
            '@typescript-eslint/no-explicit-any': 'error',
            'max-lines-per-function': ['error', 50],
        },
    },
];
