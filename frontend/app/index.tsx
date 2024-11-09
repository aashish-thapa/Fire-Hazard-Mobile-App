import React, { useEffect, useState } from 'react';
import { View, Text, ImageBackground, StyleSheet, ActivityIndicator, Alert } from 'react-native';
import * as Location from 'expo-location';
import { useFonts, Poppins_400Regular, Poppins_700Bold } from '@expo-google-fonts/poppins';
import { MaterialIcons, FontAwesome5 } from '@expo/vector-icons';
import MapView, { Marker } from 'react-native-maps';
import axios from 'axios';

const WEATHER_API_KEY = '7fd3dd027fcc5a94515388ee3c06b338';
const FIRMS_API_URL = 'https://firms.modaps.eosdis.nasa.gov/api/active_fire/';

export default function HomeScreen() {
  const [location, setLocation] = useState(null);
  const [weather, setWeather] = useState(null);
  const [firePoints, setFirePoints] = useState([]);
  const [loading, setLoading] = useState(true);

  // Load custom fonts
  let [fontsLoaded] = useFonts({
    Poppins_400Regular,
    Poppins_700Bold,
  });

  useEffect(() => {
    (async () => {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Location permissions are required to display weather data.');
        setLoading(false);
        return;
      }

      let loc = await Location.getCurrentPositionAsync({});
      setLocation(loc.coords);
      fetchWeatherData(loc.coords.latitude, loc.coords.longitude);
      fetchFireData(loc.coords.latitude, loc.coords.longitude);
    })();
  }, []);

  const fetchWeatherData = async (latitude, longitude) => {
    try {
      const response = await fetch(
        `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&appid=${WEATHER_API_KEY}&units=metric`
      );
      const data = await response.json();
      setWeather(data);
    } catch (error) {
      Alert.alert("Error", "Could not retrieve weather data.");
    } finally {
      setLoading(false);
    }
  };

  // const fetchFireData = async (latitude, longitude) => {
  //   try {
  //     const response = await axios.get(
  //       `${FIRMS_API_URL}?lat=${latitude}&lon=${longitude}&radius=100`
  //     );
  //     const data = response.data;
  //     setFirePoints(data.features);
  //   } catch (error) {
  //     Alert.alert("Error", "Could not retrieve fire data.");
  //   }
  // };

  if (!fontsLoaded || loading) {
    return <ActivityIndicator size="large" color="#00ff00" style={styles.loader} />;
  }

  return (
    <ImageBackground source={require('../assets/images/background.jpg')} style={styles.background}>
      <View style={styles.overlay}>
        <Text style={styles.appName}>Fire Hazard Prediction</Text>

        {location && weather ? (
          <View style={styles.weatherContainer}>
            <Text style={styles.weatherTitle}>Current Weather</Text>
            
            <View style={styles.weatherInfo}>
              <MaterialIcons name="location-on" size={24} color="white" />
              <Text style={styles.weatherText}>{weather?.name || 'Location Unavailable'}</Text>
            </View>

            <View style={styles.weatherInfo}>
              <FontAwesome5 name="temperature-low" size={24} color="white" />
              <Text style={styles.weatherText}>
                Temperature: {weather?.main?.temp ?? '--'}°C
              </Text>
            </View>

            <View style={styles.weatherInfo}>
              <MaterialIcons name="water-drop" size={24} color="white" />
              <Text style={styles.weatherText}>
                Humidity: {weather?.main?.humidity ?? '--'}%
              </Text>
            </View>

            <View style={styles.weatherInfo}>
              <MaterialIcons name="" size={24} color="white" />
              <Text style={styles.weatherText}>
                Wind Speed: {weather?.wind?.speed ?? '--'} m/s
              </Text>
            </View>
          </View>
        ) : (
          <Text style={styles.loadingText}>Fetching location and weather data...</Text>
        )}

        {/* {location && (
          <View style={styles.mapContainer}>
            <MapView
              style={styles.map}
              region={{
                latitude: location.latitude,
                longitude: location.longitude,
                latitudeDelta: 0.0922,
                longitudeDelta: 0.0421,
              }}
            >
              {firePoints.map((fire, index) => (
                <Marker
                  key={index}
                  coordinate={{
                    latitude: fire.geometry.coordinates[1],
                    longitude: fire.geometry.coordinates[0],
                  }}
                  title={`Fire Point ${index + 1}`}
                  description={`Confidence: ${fire.properties.confidence}`}
                />
              ))}
            </MapView>
          </View>
        )} */}
      </View>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  background: {
    flex: 1,
    resizeMode: 'cover',
  },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.6)', // Dark overlay for readability
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  appName: {
    marginTop: 0,
    fontSize: 28,
    fontFamily: 'Poppins_700Bold',
    color: '#ffffff',
    marginBottom: 30,
    textAlign: 'center',
  },
  weatherContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 10,
    padding: 20,
    width: '90%',
    alignItems: 'center',
  },
  weatherTitle: {
    fontSize: 24,
    fontFamily: 'Poppins_700Bold',
    color: '#ffffff',
    marginBottom: 15,
  },
  weatherInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 5,
  },
  weatherText: {
    fontSize: 18,
    fontFamily: 'Poppins_400Regular',
    color: '#ffffff',
    marginLeft: 10,
  },
  loadingText: {
    fontSize: 18,
    fontFamily: 'Poppins_400Regular',
    color: '#ffffff',
  },
  mapContainer: {
    width: '100%',
    height: 300,
    marginTop: 20,
  },
  map: {
    flex: 1,
    borderRadius: 10,
  },
});
