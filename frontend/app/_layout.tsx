// app/_layout.tsx
import { Tabs } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';

export default function Layout() {
  return (
    <Tabs>
      <Tabs.Screen 
        name="index" 
        options={{
          title: 'FIRE HAZARD APP',
          tabBarIcon: () => <MaterialIcons name="home" size={24} color="black" />,
        }} 
      />
      <Tabs.Screen 
        name="prediction" 
        options={{
          title: 'Predict',
          tabBarIcon: () => <MaterialIcons name="fire-extinguisher" size={24} color="black" />,
        }} 
      />
    </Tabs>
  );
}
