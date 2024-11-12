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

  // Function to fetch fire data from XWeather
  const fetchFireData = async () => {
    const client_id = 'gzSR9NjGiYdzGzItC4BpE';
    const client_secret = 'UdZ2WV8o7vb55UG4Jd4CIExUHfXm8CyMFaWu454F';
    const p = '39406'; // Example ZIP Code
    const url = `https://data.api.xweather.com/fires/closest?p=${p}&client_id=${client_id}&client_secret=${client_secret}`;

    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        const fires = data.response
          ? data.response.map((fire: any) => ({
              id: fire.id,
              latitude: fire.loc.lat,
              longitude: fire.loc.long,
            }))
          : [];
        setFireLocations(fires);
      } else {
        console.error(`Error: ${response.status} - ${response.statusText}`);
      }
    } catch (error) {
      console.error('Error fetching fire data:', error);
    }
  };

  useEffect(() => {
    (async () => {
      // Request location permissions
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Location permissions are required to display your location.');
        setLoading(false);
        return;
      }

      // Get current location
      let loc = await Location.getCurrentPositionAsync({});
      setLocation(loc.coords);

      // Fetch fire data after location retrieval
      await fetchFireData();
      setLoading(false);
    })();
  }, []);

  if (loading) {
    return <ActivityIndicator size="large" color="#FF5733" style={styles.loader} />;
  }

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

  const getPrediction = (latitude: number, longitude: number) => {
    fetchPredictionData();
    const points = [];
    const numPoints = 8; // Number of points for the shape
    
    for (let i = 0; i < numPoints; i++) {
      
      const angle = (i * (360 / numPoints)) + Math.random() * 20 - 10; // Base angle plus small randomness
      const distance = 0.005 + Math.random() * 0.005; // Base distance with random variance
      const latOffset = distance * Math.cos((angle * Math.PI) / 180);
      const lonOffset = distance * Math.sin((angle * Math.PI) / 180);
      points.push({
        latitude: latitude + latOffset,
        longitude: longitude + lonOffset,
      });
    }

    return points;
  };

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
            coordinate={{
              latitude: location.latitude,
              longitude: location.longitude,
            }}
            title="You are here"
          />
        )}

        {/* Display markers and amoeba shapes around each fire location */}
        {fireLocations.map((fire) => (
          <React.Fragment key={fire.id}>
            <Marker
              coordinate={{
                latitude: fire.latitude,
                longitude: fire.longitude,
              }}
              title={`Fire ID: ${fire.id}`}
            >
              <Image source={require('../assets/images/fire.png')} style={styles.fireIcon} />
            </Marker>
            <Polygon
              coordinates={getPrediction(fire.latitude, fire.longitude)}
              fillColor="rgba(255, 69, 0, 0.4)"
              strokeColor="rgba(255, 69, 0, 0.8)"
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
