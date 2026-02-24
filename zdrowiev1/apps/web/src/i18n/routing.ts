import { defineRouting } from 'next-intl/routing';

export const routing = defineRouting({
    // A list of all locales that are supported
    locales: ['pl', 'en'],

    // Used when no locale matches
    defaultLocale: 'pl',

    // optionally hide default locale prefix
    localePrefix: 'as-needed'
});

export type Locale = (typeof routing.locales)[number];
