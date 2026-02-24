import { useTranslations } from 'next-intl';

export default function TermsOfServicePage() {
    const t = useTranslations('Terms');

    return (
        <div className="max-w-3xl mx-auto space-y-8 animate-in fade-in duration-500 pb-16">
            <div>
                <h1 className="text-3xl font-bold tracking-tight text-foreground">{t('terms_title') || 'Regulamin Świadczenia Usług (Terms of Service)'}</h1>
                <p className="text-muted-foreground mt-2">Ostatnia aktualizacja: 24 października 2026</p>
            </div>

            <section className="glass-card p-6 space-y-4 text-foreground">
                <h2 className="text-xl font-semibold border-b border-border/50 pb-2">1. Postanowienia ogólne</h2>
                <p>
                    Rozpoczęcie korzystania z serwisu Zdrowie App oznacza pełną akceptację
                    niniejszego regulaminu. Aplikacja nie służy jako urządzenie medyczne a jej
                    wyniki mają charakter analityczno-informacyjny.
                </p>
            </section>

            <section className="glass-card p-6 space-y-4 text-foreground">
                <h2 className="text-xl font-semibold border-b border-border/50 pb-2">2. Wyłączenie odpowiedzialności medycznej</h2>
                <p className="p-4 bg-status-warning/10 text-status-warning rounded-lg border border-status-warning/20">
                    <strong>WAŻNE:</strong> Diagnostyka AI (Triage) oraz moduł OCR mają charakter wyłącznie informacyjny.
                    Zawsze konsultuj swoje objawy z certyfikowanym lekarzem pierwszej pomocy. Platforma
                    nie bierze odpowiedzialności za podjęte przez Ciebie decyzje zdrowotne.
                </p>
            </section>

            <section className="glass-card p-6 space-y-4 text-foreground">
                <h2 className="text-xl font-semibold border-b border-border/50 pb-2">3. Usługi Subskrypcyjne</h2>
                <p>
                    Dostęp do funkcji premium odbywa się poprzez odnawialną subskrypcję. Użytkownik może anulować ją w każdym
                    momencie z zachowaniem dostępu do końca trwającego okresu billingowego.
                </p>
            </section>
        </div>
    );
}
