/* apps/web/src/app/(dashboard)/reports/page.tsx */
'use client';

import styles from '@/styles/Feature.module.css';

export default function ReportsPage() {
    return (
        <div className={styles.container}>
            <header className={styles.header}>
                <h1 className={styles.title}>Raporty i Eksport</h1>
                <p className={styles.subtitle}>Generuj podsumowania PDF i eksportuj swoje dane do lekarza.</p>
            </header>

            <div className={styles.card}>
                <div className="flex flex-col items-center justify-center py-20 text-center">
                    <span className="text-6xl mb-6">PDF</span>
                    <h2 className={styles.sectionTitle}>Brak wygenerowanych raportów</h2>
                    <p className="text-color-gray-500 mb-8 max-w-md mx-auto">Twoja historia raportów jest obecnie pusta. Wygeneruj swój pierwszy raport PDF, aby zobaczyć trendy i wyniki w profesjonalnym formacie.</p>
                    <button className={`${styles.button} ${styles.primaryButton}`}>
                        Generuj nowy raport (PDF)
                    </button>
                </div>
            </div>
        </div>
    );
}
