// app/_layout.tsx
import { Tabs } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { AntDesign } from '@expo/vector-icons';

export default function Layout() {
  return (
    <Tabs>
      <Tabs.Screen 
        name="index" 
        options={{
          title: 'HOME',
          tabBarIcon: () => <MaterialIcons name="home" size={24} color="black" />,
        }} 
      />
      <Tabs.Screen 
        name="prediction" 
        options={{
          title: 'Predict',
          tabBarIcon: () => <MaterialIcons name="local-fire-department" size={24} color="black" />,
        }} 
      />
      <Tabs.Screen
        name="news"
        options={{
          title: 'News & Safety',
          tabBarIcon: () => <MaterialIcons name="newspaper" size={24} color="black" />,
        }} 
      />
      <Tabs.Screen 
        name="aboutus" 
        options={{
          title: 'About Us',
          tabBarIcon: () => <AntDesign name="team" size={24} color="black" />,
        }} 
      />
    </Tabs>
  );
}
