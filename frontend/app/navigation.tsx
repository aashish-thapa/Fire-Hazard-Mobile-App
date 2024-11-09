// app/_layout.tsx
import { Tabs } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons'; // Import MaterialIcons

export default function Layout() {
  return (
    <Tabs>
      <Tabs.Screen 
        name="index" 
        options={{
          title: 'Home',
          tabBarIcon: () => <MaterialIcons name="home" size={24} color="black" />, // Use MaterialIcons for "home" icon
        }} 
      />
      {/* You can add additional tabs here, e.g., Settings, Profile, etc. */}
    </Tabs>
  );
}
