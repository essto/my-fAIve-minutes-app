'use client';

import { AnimatePresence, motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { useTranslations } from 'next-intl';

import { useOnboarding } from './useOnboarding';
import { ProfileStep } from './steps/ProfileStep';
import { GoalsStep } from './steps/GoalsStep';
import { DevicesStep } from './steps/DevicesStep';

export const OnboardingWizard = () => {
    const router = useRouter();
    const t = useTranslations('Onboarding');
    const { step, data, nextStep, prevStep, updateProfile, updateGoals, updateDevices } = useOnboarding();
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleComplete = async (devicesData: any) => {
        updateDevices(devicesData);
        setIsSubmitting(true);

        try {
            const finalData = { ...data, devices: devicesData };

            // In a real scenario, this would likely be an API call to save user profile.
            const res = await fetch('/api/user/onboarding', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(finalData),
            });

            if (res.ok) {
                localStorage.setItem('onboarding_completed', 'true');
                router.push('/dashboard');
            } else {
                console.error('Failed to submit onboarding');
                // You could add error handling here
            }
        } catch (err) {
            console.error(err);
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center p-4 bg-background w-full">
            <motion.div
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.5, type: 'spring' }}
                className="w-full max-w-lg glass-card p-8 md:p-10 relative overflow-hidden"
            >
                {/* Premium Gradient Top Border */}
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-brand to-brand-light"></div>

                {/* Progress Indicators */}
                <div className="flex justify-between items-center mb-8">
                    <div className="flex gap-2">
                        {[1, 2, 3].map((i) => (
                            <div
                                key={i}
                                className={`h-2 rounded-full transition-all duration-300 ${i === step ? 'w-8 bg-brand' : i < step ? 'w-4 bg-brand/50' : 'w-4 bg-neutral-bg3'
                                    }`}
                            />
                        ))}
                    </div>
                    <div className="text-sm font-medium text-muted-foreground">
                        {step} / 3
                    </div>
                </div>

                <div className="relative">
                    <AnimatePresence mode="wait">
                        {step === 1 && (
                            <motion.div key="step1">
                                <ProfileStep
                                    defaultValues={data.profile}
                                    onNext={(d) => { updateProfile(d); nextStep(); }}
                                />
                            </motion.div>
                        )}
                        {step === 2 && (
                            <motion.div key="step2">
                                <GoalsStep
                                    defaultValues={data.goals}
                                    onNext={(d) => { updateGoals(d); nextStep(); }}
                                    onBack={prevStep}
                                />
                            </motion.div>
                        )}
                        {step === 3 && (
                            <motion.div key="step3">
                                <DevicesStep
                                    defaultValues={data.devices}
                                    onComplete={handleComplete}
                                    onBack={prevStep}
                                    isLoading={isSubmitting}
                                />
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </motion.div>
        </div>
    );
};
