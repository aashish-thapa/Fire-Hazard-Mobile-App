import React, { useEffect, useState } from 'react';
import { StyleSheet, View, ActivityIndicator, Alert, Image, Button } from 'react-native';
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
    const client_id = '';
    const client_secret = '';
    const p = '39406';
    const url = `https://data.api.xweather.com/fires/closest?p=${p}&client_id=${client_id}&client_secret=${client_secret}`;

    try {
      const response = await fetch(url, { method: 'GET', headers: { 'Content-Type': 'application/json' } });
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

  const fetchData = () => {
    const inputData = {
      data: Array(32).fill(
        Array(32).fill([
          0.0, 180.0, 5.0, 275.0, 290.0, 0.01, 1.5, 0.0, 0.6, 2.0, 40.0, -0.1
        ])
      )
    };

    fetch('http://{ip_adress}:8000/predict/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(inputData)
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => console.log('Response:', data))
    .catch(error => console.error('Error:', error));
  };
  // this is the random function that is predicting the fire prone area but use fetchdata function if u want to work on it
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

            <Polygon
              coordinates={predict(fire, 0.0032)}
              fillColor="rgba(139, 0, 0, 0.6)"
              strokeColor="rgba(139, 0, 0, 0.8)"
              strokeWidth={2}
            />

            <Polygon
              coordinates={predict(fire, 0.01)}
              fillColor="rgba(255, 69, 0, 0.3)"
              strokeColor="rgba(255, 69, 0, 0.7)"
              strokeWidth={2}
            />
          </React.Fragment>
        ))}
      </MapView>
      <Button title="Fetch Prediction Data" onPress={fetchData} />
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
