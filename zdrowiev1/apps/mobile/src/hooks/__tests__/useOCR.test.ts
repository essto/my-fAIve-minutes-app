import { renderHook, act, waitFor } from '@testing-library/react-native';
import { useOCR } from '../useOCR';
import api from '../../services/api';
import * as ImagePicker from 'expo-image-picker';

jest.mock('../../services/api');
jest.mock('expo-image-picker');

const mockApi = api as jest.Mocked<typeof api>;
const mockImagePicker = ImagePicker as jest.Mocked<typeof ImagePicker>;

describe('useOCR', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  // ----- O1.1 Analiza wyników z galerii -----
  it('O1.1: should pick image from gallery and upload for OCR analysis', async () => {
    // GIVEN: Image is selected
    mockImagePicker.launchImageLibraryAsync.mockResolvedValue({
      canceled: false,
      assets: [{ uri: 'file://fake/path.jpg', width: 100, height: 100 }],
    });

    // GIVEN: API returns extracted data
    mockApi.post.mockResolvedValue({
      data: {
        text: 'Wyniki Badania Krwi: Hemoglobina 14.5 g/dl',
        confidence: 0.95,
      },
    });

    const { result } = renderHook(() => useOCR());

    // WHEN: picking from gallery
    await act(async () => {
      await result.current.pickFromGallery();
    });

    // THEN: should set state correctly
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.ocrResult?.text).toContain('Hemoglobina');
      expect(mockApi.post).toHaveBeenCalledTimes(1);
    });
  });

  // ----- O1.2 Rezygnacja z wyboru zdjęcia -----
  it('O1.2: should not upload if user cancels gallery picker', async () => {
    mockImagePicker.launchImageLibraryAsync.mockResolvedValue({
      canceled: true,
      assets: null,
    });

    const { result } = renderHook(() => useOCR());

    await act(async () => {
      await result.current.pickFromGallery();
    });

    expect(mockApi.post).not.toHaveBeenCalled();
    expect(result.current.ocrResult).toBeNull();
  });

  // ----- O1.3 Robienie zdjęcia aparatem -----
  it('O1.3: should take photo and upload for OCR analysis', async () => {
    mockImagePicker.requestCameraPermissionsAsync.mockResolvedValue({
      status: ImagePicker.PermissionStatus.GRANTED,
      granted: true,
      canAskAgain: true,
      expires: 'never',
    });
    mockImagePicker.launchCameraAsync.mockResolvedValue({
      canceled: false,
      assets: [{ uri: 'file://fake/camera.jpg', width: 200, height: 200 }],
    });

    mockApi.post.mockResolvedValue({
      data: { text: 'Paracetamol 500mg', confidence: 0.99 },
    });

    const { result } = renderHook(() => useOCR());

    await act(async () => {
      await result.current.takePhoto();
    });

    await waitFor(() => {
      expect(result.current.ocrResult?.text).toBe('Paracetamol 500mg');
    });
  });

  // ----- O1.4 Błąd uprawnień kamery -----
  it('O1.4: should set error if camera permissions are denied', async () => {
    mockImagePicker.requestCameraPermissionsAsync.mockResolvedValue({
      status: ImagePicker.PermissionStatus.DENIED,
      granted: false,
      canAskAgain: false,
      expires: 'never',
    });

    const { result } = renderHook(() => useOCR());

    await act(async () => {
      await result.current.takePhoto();
    });

    expect(result.current.error).toBe('Aplikacja wymaga dostępu do aparatu.');
    expect(mockImagePicker.launchCameraAsync).not.toHaveBeenCalled();
  });

  // ----- O1.5 Błąd sieci podczas analizy -----
  it('O1.5: should handle API error gracefully during upload', async () => {
    mockImagePicker.launchImageLibraryAsync.mockResolvedValue({
      canceled: false,
      assets: [{ uri: 'file://fake/path.jpg', width: 100, height: 100 }],
    });

    mockApi.post.mockRejectedValue(new Error('Network upload failed'));

    const { result } = renderHook(() => useOCR());

    await act(async () => {
      await result.current.pickFromGallery();
    });

    await waitFor(() => {
      expect(result.current.error).toBe('Network upload failed');
      expect(result.current.ocrResult).toBeNull();
    });
  });
});
