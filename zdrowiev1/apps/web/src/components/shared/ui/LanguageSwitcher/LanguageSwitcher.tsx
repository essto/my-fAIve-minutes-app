'use client';

import { useLocale } from 'next-intl';
import { useRouter, usePathname } from 'next/navigation';
import { useTransition } from 'react';
import { Button } from '../Button/Button';

export const LanguageSwitcher = () => {
    const [isPending, startTransition] = useTransition();
    const router = useRouter();
    const locale = useLocale();
    const pathname = usePathname();

    const switchLanguage = (newLocale: string) => {
        startTransition(() => {
            // Create new path by replacing the current locale prefix with the new one
            const pathSegments = pathname.split('/').filter(Boolean);

            let newPathname = `/${newLocale}`;

            // If path starts with a locale, remove it
            if (['pl', 'en'].includes(pathSegments[0])) {
                pathSegments.shift();
            }

            if (pathSegments.length > 0) {
                newPathname += `/${pathSegments.join('/')}`;
            }

            router.replace(newPathname);
        });
    };

    return (
        <div className="flex gap-2 items-center">
            <div className="text-sm font-bold w-6 text-center">{locale.toUpperCase()}</div>
            <Button
                variant="outline"
                onClick={() => switchLanguage(locale === 'pl' ? 'en' : 'pl')}
                disabled={isPending}
                title="Zmień język / Change language"
            >
                {locale === 'pl' ? 'English' : 'Polski'}
            </Button>
        </div>
    );
};
