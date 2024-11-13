import React, { useEffect, useState } from 'react';
import { StyleSheet, View, ActivityIndicator, Alert, Image } from 'react-native';
import * as Location from 'expo-location';
import MapView, { Marker, Polygon } from 'react-native-maps';

interface FireLocation {
  id: string;
  latitude: number;
  longitude: number;
}

export default function PredictionScreen() {
  const [location, setLocation] = useState<{ latitude: number; longitude: number } | null>(null);
  const [fireLocations, setFireLocations] = useState<FireLocation[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchFireData = async () => {
    const client_id = 'gzSR9NjGiYdzGzItC4BpE';
    const client_secret = 'UdZ2WV8o7vb55UG4Jd4CIExUHfXm8CyMFaWu454F';
    const p = '39406';
    const url = `https://data.api.xweather.com/fires/closest?p=${p}&client_id=${client_id}&client_secret=${client_secret}`;

    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      if (response.ok) {
        const data = await response.json();
        const fires = data.response.map((fire: any) => ({
          id: fire.id,
          latitude: fire.loc.lat,
          longitude: fire.loc.long,
        }));
        setFireLocations(fires);
      } else {
        console.error(`Error: ${response.status} - ${response.statusText}`);
      }
    } catch (error) {
      console.error('Error fetching fire data:', error);
    }
  };
  // Prediction function that fetches data from the prediction URL but does not automatically call it
const fetchPredictionData = async () => {
  const inputData = {
    data: [
      [
        [
          [0.0, 180.0, 5.0, 275.0, 290.0, 0.01, 1.5, 0.0, 0.6, 2.0, 40.0, -0.1],
          // Repeat similar values for the required dimensions (32x32)
        ]
      ]
    ]
  };

  const url = 'http://34.228.190.91:8000/predict/';

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(inputData),
    });

    if (response.ok) {
      const data = await response.json();
      console.log('Prediction data:', data);
      
    } else {
      console.error(`Error: ${response.status} - ${response.statusText}`);
    }
  } catch (error) {
    console.error('Error fetching prediction data:', error);
  }
};

// To use fetchPredictionData, call it manually where needed, but it won’t run automatically on import or component load.

  const predict = (center: { latitude: number; longitude: number }, radius: number) => {
    
    const points = [];
    const sides = 40; 
    for (let i = 0; i < sides; i++) {
      const angle = (i / sides) * 2 * Math.PI;
      const randomOffset = (Math.random() * 0.0008 + 0.0002) * (Math.random() < 0.5 ? 1 : -1);
      const latOffset = Math.cos(angle) * (radius + randomOffset);
      const lngOffset = Math.sin(angle) * (radius + randomOffset);
      points.push({ latitude: center.latitude + latOffset, longitude: center.longitude + lngOffset });
    }
    return points;
  };

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
      await fetchFireData();
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
          latitude: location?.latitude || 0,
          longitude: location?.longitude || 0,
          latitudeDelta: 0.0922,
          longitudeDelta: 0.0421,
        }}
        showsUserLocation={true}
      >
        {location && (
          <Marker
            coordinate={{ latitude: location.latitude, longitude: location.longitude }}
            title="You are here"
          />
        )}

        {fireLocations.map((fire) => (
          <React.Fragment key={fire.id}>
            <Marker
              coordinate={{ latitude: fire.latitude, longitude: fire.longitude }}
              title={`Fire ID: ${fire.id}`}
            >
              <Image source={require('../assets/images/fire.png')} style={styles.fireIcon} />
            </Marker>

            {/* Small Dark Red Gradient */}
            <Polygon
              coordinates={predict(fire, 0.0032)}  // Smaller radius
              fillColor="rgba(139, 0, 0, 0.6)" // Dark red, more opaque
              strokeColor="rgba(139, 0, 0, 0.8)" // Dark red outline
              strokeWidth={2}
            />

            {/* Larger Light Red Gradient */}
            <Polygon
              coordinates={predict(fire, 0.01)}  // Larger radius
              fillColor="rgba(255, 69, 0, 0.3)" // Lighter red, semi-transparent
              strokeColor="rgba(255, 69, 0, 0.7)" // Darker red outline
              strokeWidth={2}
            />
          </React.Fragment>
        ))}
      </MapView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#2E2A2A',
  },
  map: {
    ...StyleSheet.absoluteFillObject,
  },
  loader: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  fireIcon: {
    width: 40,
    height: 40,
  },
});
