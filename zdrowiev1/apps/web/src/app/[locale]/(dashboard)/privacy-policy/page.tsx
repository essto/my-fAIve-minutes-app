import { useTranslations } from 'next-intl';

export default function PrivacyPolicyPage() {
    const t = useTranslations('Terms');

    return (
        <div className="max-w-3xl mx-auto space-y-8 animate-in fade-in duration-500 pb-16">
            <div>
                <h1 className="text-3xl font-bold tracking-tight text-foreground">{t('privacy_title') || 'Polityka prywatności (Privacy Policy)'}</h1>
                <p className="text-muted-foreground mt-2">Ostatnia aktualizacja: 24 października 2026</p>
            </div>

            <section className="glass-card p-6 space-y-4 text-foreground">
                <h2 className="text-xl font-semibold border-b border-border/50 pb-2">1. Dane osobowe i RODO (GDPR)</h2>
                <p>
                    Ochrona Twojej prywatności jest naszym priorytetem. Zbieramy jedynie dane
                    niezbędne do świadczenia usług analitycznych i monitorowania parametrów życiowych.
                    Zgodnie z RODO (GDPR) masz pełne prawo do wglądu, modyfikacji i usunięcia swoich danych w dowolnym momencie.
                </p>
            </section>

            <section className="glass-card p-6 space-y-4 text-foreground">
                <h2 className="text-xl font-semibold border-b border-border/50 pb-2">2. Kategorie przetwarzanych danych</h2>
                <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                    <li><strong className="text-foreground">Dane profilowe:</strong> Imię, wiek, płeć, waga, wzrost.</li>
                    <li><strong className="text-foreground">Dane telemetryczne:</strong> Tętno, sen, kroki pobierane z urządzeń zew.</li>
                    <li><strong className="text-foreground">Dane z AI:</strong> Historie czatów triage oraz raporty OCR (przetwarzane anonimowo).</li>
                </ul>
            </section>

            <section className="glass-card p-6 space-y-4 text-foreground">
                <h2 className="text-xl font-semibold border-b border-border/50 pb-2">3. Udostępnianie danych</h2>
                <p>
                    Zdrowie App <strong>nie sprzedaje i nigdy nie będzie sprzedawać</strong> Twoich danych osobowych.
                    Twoje dane są izolowane z użyciem protokołu RLS na poziomie bazy danych i wykorzystywane wyłącznie
                    w obrębie działania algorytmów na Twoim koncie.
                </p>
            </section>
        </div>
    );
}
