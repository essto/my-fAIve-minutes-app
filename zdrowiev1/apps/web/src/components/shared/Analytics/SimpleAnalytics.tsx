'use client';

import { useEffect } from 'react';
import { usePathname, useSearchParams } from 'next/navigation';

export function Analytics() {
    const pathname = usePathname();
    const searchParams = useSearchParams();

    useEffect(() => {
        if (pathname) {
            // W środowisku produkcyjnym można tutaj wysłać zdarzenie do Google Analytics / Mixpanel itp.
            const url = pathname + (searchParams?.toString() ? `?${searchParams.toString()}` : '');
            if (process.env.NODE_ENV === 'production') {
                console.log(`[Analytics] Pageview: ${url}`);
            }
        }
    }, [pathname, searchParams]);

    return null;
}
