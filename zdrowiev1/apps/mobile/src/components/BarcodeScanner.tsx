import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

interface BarcodeScannerProps {
  onBarcodeScanned: (data: string) => void;
  onClose: () => void;
}

export const BarcodeScanner = ({ onBarcodeScanned, onClose }: BarcodeScannerProps) => {
  const [permission, requestPermission] = useCameraPermissions();

  if (!permission) {
    return <View />;
  }

  if (!permission.granted) {
    return (
      <View className="flex-1 justify-center items-center bg-black">
        <Text className="text-white mb-4 text-center">
          Potrzebujemy dostępu do aparatu, aby skanować kody kreskowe.
        </Text>
        <TouchableOpacity className="bg-brand py-3 px-6 rounded-xl" onPress={requestPermission}>
          <Text className="text-white font-bold">Udziel dostępu</Text>
        </TouchableOpacity>
        <TouchableOpacity className="mt-4 p-2" onPress={onClose}>
          <Text className="text-white/70">Anuluj</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const handleBarcodeScanned = ({ data }: { data: string }) => {
    onBarcodeScanned(data);
  };

  return (
    <View className="flex-1 bg-black">
      <CameraView
        testID="camera-view"
        style={StyleSheet.absoluteFillObject}
        barcodeScannerSettings={{
          barcodeTypes: ['ean13', 'ean8', 'upc_e', 'upc_a', 'qr'],
        }}
        onBarcodeScanned={handleBarcodeScanned}
      >
        <View className="flex-1 justify-center items-center bg-black/60">
          {/* Viewfinder cutout frame */}
          <View
            testID="barcode-viewfinder"
            className="w-[80%] h-64 border-2 border-brand rounded-2xl bg-transparent"
          />

          <Text className="text-white mt-8 px-8 text-center text-lg font-medium">
            Nakieruj aparat na kod kreskowy
          </Text>
        </View>

        <TouchableOpacity
          className="absolute top-12 right-6 p-4 bg-black/40 rounded-full"
          onPress={onClose}
        >
          <Text className="text-white font-bold">Anuluj</Text>
        </TouchableOpacity>
      </CameraView>
    </View>
  );
};
