'use client';

import { useLocale } from 'next-intl';
import { useRouter, usePathname } from '@/i18n/routing';
import { useTransition } from 'react';
import { Button } from '../Button/Button';

export const LanguageSwitcher = () => {
    const [isPending, startTransition] = useTransition();
    const router = useRouter();
    const locale = useLocale();
    const pathname = usePathname();

    const switchLanguage = (newLocale: 'pl' | 'en') => {
        startTransition(() => {
            router.replace(pathname, { locale: newLocale });
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
