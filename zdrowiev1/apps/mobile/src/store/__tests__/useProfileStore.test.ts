import { useProfileStore } from '../useProfileStore';

describe('useProfileStore', () => {
  beforeEach(() => {
    // Reset store before each test
    const initialState = useProfileStore.getState();
    useProfileStore.setState(initialState, true);
  });

  it('P1.1: should have default initial state', () => {
    const state = useProfileStore.getState();
    expect(state.language).toBe('pl');
    expect(state.theme).toBe('dark');
    expect(state.notificationsEnabled).toBe(true);
    expect(state.connectedDevices).toEqual([]);
  });

  it('P1.2: should toggle theme', () => {
    expect(useProfileStore.getState().theme).toBe('dark');
    useProfileStore.getState().setTheme('light');
    expect(useProfileStore.getState().theme).toBe('light');
  });

  it('P1.3: should change language', () => {
    expect(useProfileStore.getState().language).toBe('pl');
    useProfileStore.getState().setLanguage('en');
    expect(useProfileStore.getState().language).toBe('en');
  });

  it('P1.4: should manage connected devices', () => {
    const device = { id: 'd1', name: 'Xiaomi Band', type: 'watch' as const };

    useProfileStore.getState().addDevice(device);
    expect(useProfileStore.getState().connectedDevices).toContainEqual(device);

    useProfileStore.getState().removeDevice('d1');
    expect(useProfileStore.getState().connectedDevices).not.toContainEqual(device);
  });
});
