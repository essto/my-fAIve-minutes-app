import { useState } from 'react';
import { useTranslations } from 'next-intl';
import { DevicesData } from '../useOnboarding';
import { motion } from 'framer-motion';

interface Props {
    defaultValues: Partial<DevicesData>;
    onComplete: (data: DevicesData) => void;
    onBack: () => void;
    isLoading?: boolean;
}

export const DevicesStep = ({ defaultValues, onComplete, onBack, isLoading }: Props) => {
    const t = useTranslations('Onboarding');
    const [selectedDevices, setSelectedDevices] = useState<string[]>(defaultValues.devices || []);

    const toggleDevice = (id: string) => {
        setSelectedDevices(prev =>
            prev.includes(id)
                ? prev.filter(d => d !== id)
                : [...prev, id]
        );
    };

    const devices = [
        { id: 'apple_health', name: 'Apple Health', icon: '🍏' },
        { id: 'garmin', name: 'Garmin Connect', icon: '⌚' },
        { id: 'oura', name: 'Oura Ring', icon: '💍' },
        { id: 'google_fit', name: 'Google Fit', icon: '🏃' },
    ];

    const handleSubmit = () => {
        onComplete({ devices: selectedDevices });
    };

    return (
        <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
        >
            <div className="text-center mb-6">
                <h2 className="text-2xl font-bold text-foreground">{t('step_3_title')}</h2>
                <p className="text-muted-foreground mt-2">{t('step_3_desc')}</p>
            </div>

            <div className="grid grid-cols-2 gap-4">
                {devices.map((device) => {
                    const isSelected = selectedDevices.includes(device.id);
                    return (
                        <button
                            key={device.id}
                            type="button"
                            onClick={() => toggleDevice(device.id)}
                            className={`flex flex-col items-center justify-center p-6 rounded-xl border transition-all ${isSelected
                                    ? 'bg-brand/10 border-brand shadow-glow scale-[1.02]'
                                    : 'bg-neutral-bg2 border-border hover:border-brand/40'
                                }`}
                        >
                            <span className="text-4xl mb-3">{device.icon}</span>
                            <span className="font-semibold text-foreground text-center">{device.name}</span>
                        </button>
                    );
                })}
            </div>

            <div className="mt-8 flex gap-4">
                <button
                    type="button"
                    title="Cofnij"
                    aria-label="back"
                    onClick={onBack}
                    className="flex-1 py-3.5 rounded-xl text-muted-foreground bg-neutral-bg3 hover:bg-neutral-bg2 transition-colors font-medium border border-border"
                    disabled={isLoading}
                >
                    {t('back')}
                </button>
                <button
                    type="button"
                    aria-label="finish"
                    onClick={handleSubmit}
                    className="flex-[2] bg-brand hover:bg-brand-hover text-white font-semibold py-3.5 rounded-xl transition-all shadow-glow hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50"
                    disabled={isLoading}
                >
                    {isLoading ? '...' : t('finish')}
                </button>
            </div>
        </motion.div>
    );
};
