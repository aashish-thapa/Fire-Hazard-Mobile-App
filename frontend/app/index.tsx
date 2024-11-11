import React, { useEffect, useState } from 'react';
import { View, Text, Alert, StyleSheet, ActivityIndicator, TextInput, TouchableOpacity } from 'react-native';
import * as Location from 'expo-location';
import * as Notifications from 'expo-notifications';
import { useFonts, Poppins_400Regular, Poppins_700Bold } from '@expo-google-fonts/poppins';
import { MaterialIcons, FontAwesome5 } from '@expo/vector-icons';
import MapView, { Marker } from 'react-native-maps';
import { LinearGradient } from 'expo-linear-gradient';

const WEATHER_API_KEY = '7fd3dd027fcc5a94515388ee3c06b338';
const IP_ADDRESS = "10.0.0.183";
export default function HomeScreen() {
  const [location, setLocation] = useState({
    latitude:0,
    longitude:0
  });
  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(true);
  const [email, setEmail] = useState('');
  const [isFormVisible, setIsFormVisible] = useState(false); // Track form visibility

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

      // Get location
      let loc = await Location.getCurrentPositionAsync({});

      

      setLocation({
        latitude: loc.coords.latitude,
        longitude: loc.coords.longitude
      });

      fetchWeatherData(loc.coords.latitude, loc.coords.longitude);
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

  const handleSubmitEmail = async () => {
    if (!email || !location) {
      Alert.alert('Error', 'Please enter a valid email address and make sure location is available.');
      return;
    }
  
    // Generate a unique user ID
    const userId = generateUniqueId();
  
    // Prepare the data to send to the backend
    const userData = {
      userId: userId,
      email: email,
      latitude: location.latitude,
      longitude: location.longitude,
    };
  
    // Send the data to the backend
    try {
      const response = await fetch(`http://${IP_ADDRESS}:3000/storeUserData`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      
  
      const data = await response.json();
      if (data.success) {
        Alert.alert('Success', 'Your information has been submitted successfully.');
        setIsFormVisible(false); // Close the form after submission
      } else {
        Alert.alert('Error', data.message || 'Could not save data to the database.');
      }
    } catch (error) {
      Alert.alert('Error', 'An error occurred while saving data.');
      console.error(error);
    }
  };
  
  // Function to generate a unique ID (simple example)
  const generateUniqueId = () => {
    return new Date().getTime().toString();  // Generate unique ID based on timestamp
  };
  

  const handleCancelForm = () => {
    setIsFormVisible(false); // Close the form
  };

  if (!fontsLoaded || loading) {
    return <ActivityIndicator size="large" color="#00ff00" style={styles.loader} />;
  }

  return (
    <LinearGradient colors={['#0F2027', '#203A43', '#2C5364']} style={styles.background}>
      <View style={styles.overlay}>
        <Text style={styles.appName}>Fire Hazard Prediction</Text>
        
        {/* Email Form Popup */}
        {isFormVisible && (
          <View style={styles.formContainer}>
            <Text style={styles.formTitle}>Enter Your Email</Text>
            <TextInput
              style={styles.input}
              placeholder="Enter your email here"
              placeholderTextColor="#aaa"
              value={email}
              onChangeText={setEmail}
            />
            <View style={styles.formButtons}>
              <TouchableOpacity style={styles.button} onPress={handleSubmitEmail}>
                <Text style={styles.buttonText}>Submit</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.button} onPress={handleCancelForm}>
                <Text style={styles.buttonText}>Cancel</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Button to show email form */}
        {!isFormVisible && (
          <TouchableOpacity
            style={styles.showFormButton}
            onPress={() => setIsFormVisible(true)}
          >
            <Text style={styles.showFormButtonText}>Enable Fire Alert</Text>
          </TouchableOpacity>
        )}

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
    backgroundColor: '#330000',
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
  formContainer: {
    backgroundColor: 'rgba(255, 99, 71, 0.8)',
    padding: 20,
    borderRadius: 15,
    width: 300,
    alignItems: 'center',
    marginTop: 10,
    shadowColor: '#8B0000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 4.65,
    elevation: 8,
  },
  formTitle: {
    fontSize: 20,
    fontFamily: 'Poppins_700Bold',
    color: '#ffffff',
    marginBottom: 10,
  },
  input: {
    height: 40,
    width: '100%',
    borderColor: '#fff',
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 10,
    marginBottom: 20,
    color: '#fff',
  },
  formButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
  },
  button: {
    backgroundColor: '#FF6347',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  buttonText: {
    color: '#fff',
    fontFamily: 'Poppins_400Regular',
  },
  showFormButton: {
    backgroundColor: '#FF6347',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginTop: 0,
    marginBottom:15,
    marginLeft:230,
    marginRight: 0,
  },
  showFormButtonText: {
    color: '#fff',
    fontFamily: 'Poppins_700Bold',
  },
});

