import React, { useEffect, useState } from 'react';
import { StyleSheet, View, ActivityIndicator, Alert } from 'react-native';
import * as Location from 'expo-location';
import MapView, { Marker } from 'react-native-maps';

export default function PredictionScreen() {
  const [location, setLocation] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Location permissions are required to display your location.');
        setLoading(false);
        return;
      }

      let loc = await Location.getCurrentPositionAsync({});
      setLocation(loc.coords);
      setLoading(false);
    })();
  }, []);

  if (loading) {
    return <ActivityIndicator size="large" color="#FF5733" style={styles.loader} />;
  }

  return (
    <View style={styles.container}>
      <MapView
        style={styles.map}
        initialRegion={{
          latitude: location.latitude,
          longitude: location.longitude,
          latitudeDelta: 0.0922,
          longitudeDelta: 0.0421,
        }}
        showsUserLocation={true}  // Show user location marker
      >
        <Marker
          coordinate={{
            latitude: location.latitude,
            longitude: location.longitude,
          }}
          title="You are here"
        />
      </MapView>
    </View>
  );
}
const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: '#2E2A2A', // Optional: Red-themed background color to align with fire app theme
    },
    map: {
      ...StyleSheet.absoluteFillObject,
    },
    loader: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
    },
  });
  