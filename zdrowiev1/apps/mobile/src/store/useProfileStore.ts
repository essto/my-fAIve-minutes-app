import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

export type DeviceType = 'watch' | 'scale' | 'other';

export interface ConnectedDevice {
  id: string;
  name: string;
  type: DeviceType;
}

export interface ProfileState {
  language: 'pl' | 'en';
  theme: 'light' | 'dark';
  notificationsEnabled: boolean;
  connectedDevices: ConnectedDevice[];
}

export interface ProfileActions {
  setLanguage: (lang: 'pl' | 'en') => void;
  setTheme: (theme: 'light' | 'dark') => void;
  toggleNotifications: () => void;
  addDevice: (device: ConnectedDevice) => void;
  removeDevice: (id: string) => void;
  reset: () => void;
}

export type ProfileStore = ProfileState & ProfileActions;

const initialState: ProfileState = {
  language: 'pl',
  theme: 'dark',
  notificationsEnabled: true,
  connectedDevices: [],
};

export const useProfileStore = create<ProfileStore>()(
  persist(
    (set) => ({
      ...initialState,
      setLanguage: (language) => set({ language }),
      setTheme: (theme) => set({ theme }),
      toggleNotifications: () =>
        set((state) => ({ notificationsEnabled: !state.notificationsEnabled })),
      addDevice: (device) =>
        set((state) => {
          // Prevent duplicates
          if (state.connectedDevices.some((d) => d.id === device.id)) {
            return state;
          }
          return { connectedDevices: [...state.connectedDevices, device] };
        }),
      removeDevice: (id) =>
        set((state) => ({
          connectedDevices: state.connectedDevices.filter((d) => d.id !== id),
        })),
      reset: () => set(initialState),
    }),
    {
      name: 'profile-storage',
      storage: createJSONStorage(() => AsyncStorage),
    },
  ),
);
