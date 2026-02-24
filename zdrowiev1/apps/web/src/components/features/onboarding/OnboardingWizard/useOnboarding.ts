import { useState } from 'react';
import { z } from 'zod';

export const ProfileSchema = z.object({
    name: z.string().min(1, 'name_required'),
    age: z.number().min(1, 'age_required'),
    height: z.number().min(1, 'height_required'),
    weight: z.number().min(1, 'weight_required'),
});

export const GoalsSchema = z.object({
    goal: z.enum(['lose_weight', 'gain_weight', 'maintain_weight']),
});

export const DevicesSchema = z.object({
    devices: z.array(z.string()).optional(),
});

export type ProfileData = z.infer<typeof ProfileSchema>;
export type GoalsData = z.infer<typeof GoalsSchema>;
export type DevicesData = z.infer<typeof DevicesSchema>;

export interface OnboardingData {
    profile: Partial<ProfileData>;
    goals: Partial<GoalsData>;
    devices: Partial<DevicesData>;
}

export const useOnboarding = () => {
    const [step, setStep] = useState(1);
    const [data, setData] = useState<OnboardingData>({
        profile: {},
        goals: {},
        devices: {},
    });

    const nextStep = () => setStep((s) => Math.min(3, s + 1));
    const prevStep = () => setStep((s) => Math.max(1, s - 1));

    const updateProfile = (profileData: Partial<ProfileData>) => {
        setData((prev) => ({ ...prev, profile: { ...prev.profile, ...profileData } }));
    };

    const updateGoals = (goalsData: Partial<GoalsData>) => {
        setData((prev) => ({ ...prev, goals: { ...prev.goals, ...goalsData } }));
    };

    const updateDevices = (devicesData: Partial<DevicesData>) => {
        setData((prev) => ({ ...prev, devices: { ...prev.devices, ...devicesData } }));
    };

    return {
        step,
        data,
        nextStep,
        prevStep,
        updateProfile,
        updateGoals,
        updateDevices,
    };
};
