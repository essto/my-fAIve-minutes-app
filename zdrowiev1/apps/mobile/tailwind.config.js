/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./App.{js,jsx,ts,tsx}', './src/**/*.{js,jsx,ts,tsx}'],
  presets: [require('nativewind/preset')],
  theme: {
    extend: {
      colors: {
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        card: 'hsl(var(--card))',
        'card-foreground': 'hsl(var(--card-foreground))',
        border: 'hsl(var(--border))',
        brand: {
          DEFAULT: '#8251EE',
          light: '#9D75F0',
          dark: '#6730E7',
          hover: '#733CE6',
        },
        neutral: {
          bg1: '#121212',
          bg2: '#18181A',
          bg3: '#222224',
          bg4: '#2A2A2D',
        },
      },
    },
  },
  plugins: [],
};
