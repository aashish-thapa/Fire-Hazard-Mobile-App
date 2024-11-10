import React from 'react';
import { View, Text, Image, StyleSheet, ScrollView, Animated } from 'react-native';
import { useFonts, Poppins_400Regular, Poppins_700Bold } from '@expo-google-fonts/poppins';

export default function AboutUsScreen() {
  const scrollY = new Animated.Value(0);
  
  let [fontsLoaded] = useFonts({
    Poppins_400Regular,
    Poppins_700Bold,
  });

  if (!fontsLoaded) {
    return null; // Add a loading spinner if needed
  }

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.contentContainer}
      onScroll={Animated.event(
        [{ nativeEvent: { contentOffset: { y: scrollY } } }],
        { useNativeDriver: false }
      )}
    >
      <View style={styles.headerContainer}>
        <Animated.Text style={styles.headerText}>About Us</Animated.Text>
      </View>

      <View style={styles.cardContainer}>
        <View style={styles.card}>
          <Image source={require('../assets/images/team.jpg')} style={styles.cardImage} />
          <Text style={styles.cardTitle}>Our Mission</Text>
          <Text style={styles.cardDescription}>
            We aim to protect communities by providing real-time fire predictions and alerts, harnessing cutting-edge technology and AI.
          </Text>
        </View>

        <View style={styles.card}>
          <Image source={require('../assets/images/technology.jpeg')} style={styles.cardImage} />
          <Text style={styles.cardTitle}>Technology</Text>
          <Text style={styles.cardDescription}>
            Our platform utilizes NASA APIs, predictive models, and advanced analytics to track and predict fire hazards.
          </Text>
        </View>

        <View style={styles.card}>
          <Image source={require('../assets/images/impact.jpeg')} style={styles.cardImage} />
          <Text style={styles.cardTitle}>Impact</Text>
          <Text style={styles.cardDescription}>
            We strive to make a positive impact by ensuring safer environments and proactive response to fire hazards.
          </Text>
        </View>
      </View>
    </ScrollView>
  );
}
const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: '#2E2A2A', // Dark background for contrast
    },
    contentContainer: {
      alignItems: 'center',
      paddingVertical: 20,
    },
    headerContainer: {
      width: '100%',
      paddingVertical: 30,
      backgroundColor: 'linear-gradient(45deg, #FF5733, #C70039, #900C3F)',
      alignItems: 'center',
      borderBottomLeftRadius: 50,
      borderBottomRightRadius: 50,
      shadowColor: '#FF5733',
      shadowOffset: { width: 0, height: 10 },
      shadowOpacity: 0.5,
      shadowRadius: 20,
    },
    headerText: {
      fontSize: 36,
      fontFamily: 'Poppins_700Bold',
      color: '#FFF',
      textAlign: 'center',
      marginVertical: 20,
      textShadowColor: 'rgba(255, 87, 51, 0.6)',
      textShadowOffset: { width: 2, height: 4 },
      textShadowRadius: 10,
    },
    cardContainer: {
      width: '90%',
      marginTop: 20,
      paddingVertical: 20,
    },
    card: {
      backgroundColor: '#1B1B1B',
      borderRadius: 20,
      padding: 20,
      marginVertical: 15,
      alignItems: 'center',
      shadowColor: '#FF5733',
      shadowOffset: { width: 0, height: 5 },
      shadowOpacity: 0.3,
      shadowRadius: 10,
      elevation: 8,
    },
    cardImage: {
      width: '100%',
      height: 150,
      borderRadius: 15,
      marginBottom: 15,
    },
    cardTitle: {
      fontSize: 24,
      fontFamily: 'Poppins_700Bold',
      color: '#FF5733',
      textAlign: 'center',
      marginVertical: 5,
    },
    cardDescription: {
      fontSize: 16,
      fontFamily: 'Poppins_400Regular',
      color: '#EAEAEA',
      textAlign: 'center',
      lineHeight: 22,
    },
  });
  