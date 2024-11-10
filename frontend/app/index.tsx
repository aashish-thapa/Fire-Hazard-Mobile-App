import React, { useEffect, useState } from 'react';
import { View, Text, Alert, StyleSheet, ActivityIndicator } from 'react-native';
import * as Location from 'expo-location';
import * as Notifications from 'expo-notifications';
import { useFonts, Poppins_400Regular, Poppins_700Bold } from '@expo-google-fonts/poppins';
import { MaterialIcons, FontAwesome5 } from '@expo/vector-icons';
import MapView, { Marker } from 'react-native-maps';
import { LinearGradient } from 'expo-linear-gradient';

const WEATHER_API_KEY = '7fd3dd027fcc5a94515388ee3c06b338';

export default function HomeScreen() {
  const [location, setLocation] = useState(null);
  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(true);

  let [fontsLoaded] = useFonts({
    Poppins_400Regular,
    Poppins_700Bold,
  });

  useEffect(() => {
    // Request location permission
    (async () => {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Location permissions are required to display weather data.');
        setLoading(false);
        return;
      }

      // Request notification permission
      // await registerForPushNotifications();

      // Get location
      let loc = await Location.getCurrentPositionAsync({});
      setLocation(loc.coords);
      fetchWeatherData(loc.coords.latitude, loc.coords.longitude);
    })();
  }, []);

  // const registerForPushNotifications = async () => {
  //   const { status } = await Notifications.requestPermissionsAsync();
  //   if (status !== 'granted') {
  //     Alert.alert('Permission Denied', 'Push notification permissions are required.');
  //     return;
  //   }

  //   const token = (await Notifications.getExpoPushTokenAsync()).data;
  //   console.log("Push Notification Token: ", token);
  // };

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

  if (!fontsLoaded || loading) {
    return <ActivityIndicator size="large" color="#00ff00" style={styles.loader} />;
  }

  return (
    <LinearGradient colors={['#0F2027', '#203A43', '#2C5364']} style={styles.background}>
      <View style={styles.overlay}>
        <Text style={styles.appName}>Fire Hazard Prediction</Text>

        {location && weather ? (
          <View style={styles.card}>
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
              <MaterialIcons name="air" size={24} color="white" />
              <Text style={styles.weatherText}>
                Wind Speed: {weather?.wind?.speed ?? '--'} m/s
              </Text>
            </View>
          </View>
        ) : (
          <Text style={styles.loadingText}>Fetching location and weather data...</Text>
        )}

        <View style={styles.mapContainer}>
          <MapView
            style={styles.map}
            region={{
              latitude: location?.latitude || 37.78825,
              longitude: location?.longitude || -122.4324,
              latitudeDelta: 0.0922,
              longitudeDelta: 0.0421,
            }}
          >
            {location && (
              <Marker
                coordinate={{
                  latitude: location.latitude,
                  longitude: location.longitude,
                }}
                title="You are here"
                description="Your current location"
              />
            )}
          </MapView>
        </View>
      </View>
    </LinearGradient>
  );
}

// Rest of your styles...

const styles = StyleSheet.create({
  background: {
    flex: 1,
     backgroundColor: '#330000'
  },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(255, 69, 0, 0.4)',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  appName: {
    fontSize: 28,
    fontFamily: 'Poppins_700Bold',
    color: '#ffffff',
    marginBottom: 20,
    textAlign: 'center',
    textShadowColor: 'rgba(255, 69, 0, 0.5)',
    textShadowOffset: { width: 2, height: 2 },
    textShadowRadius: 5,
  },
  card: {
    backgroundColor: 'rgba(255, 99, 71, 0.3)',
    borderRadius: 15,
    padding: 20,
    width: '100%',
    alignItems: 'center',
    marginBottom: 20,
    shadowColor: '#8B0000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 4.65,
    elevation: 8,
  },
  weatherTitle: {
    fontSize: 22,
    fontFamily: 'Poppins_700Bold',
    color: '#ffffff',
    marginBottom: 10,
  },
  weatherInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 8,
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
    borderRadius: 15,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 4.65,
    elevation: 8,
  },
  map: {
    flex: 1,
  },
});
