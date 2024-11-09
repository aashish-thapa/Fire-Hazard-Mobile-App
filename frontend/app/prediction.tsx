// app/prediction.tsx
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

export default function PredictionScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.header}>Prediction Screen</Text>
      <Text>This is where the fire spread prediction model will be implemented.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    fontSize: 24,
    fontWeight: 'bold',
  },
});
