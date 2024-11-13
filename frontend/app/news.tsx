import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Modal } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

export default function NewsSafetyScreen() {
  const [news, setNews] = useState([]);
  const [error, setError] = useState('');
  const [modalVisible, setModalVisible] = useState(false);
  const [selectedArticle, setSelectedArticle] = useState(null);

  // Function to fetch news
  const fetchWeatherNews = async () => {
    try {
      const response = await fetch(
        "https://newsapi.org/v2/everything?q=weather&from=2024-10-12&sortBy=publishedAt&apiKey=" //enter your key
      );
      
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'ok' && data.articles) {
          setNews(data.articles);  // Correctly set the state to the articles array
        } else {
          setError('No news available.');
        }
      } else {
        setError('Failed to fetch news. Please try again later.');
      }
    } catch (err) {
      setError('Failed to fetch news');
    }
  };

  useEffect(() => {
    // Fetch news when component mounts
    fetchWeatherNews();

    // Set an interval to refresh the page every 5 seconds
    const interval = setInterval(() => {
      fetchWeatherNews();
    }, 5000);

    // Clean up interval when component unmounts
    return () => clearInterval(interval);
  }, []); // Empty dependency array ensures this runs once on mount

  const handleNewsItemPress = (article) => {
    setSelectedArticle(article);  // Store the selected article
    setModalVisible(true);  // Open the modal
  };

  return (
    <LinearGradient colors={['#ff6a00', '#ff3a00']} style={styles.container}>
      <View style={styles.newsContainer}>
        <Text style={styles.sectionTitle}>Latest Weather Hazard News</Text>
        {error ? (
          <Text style={styles.errorText}>{error}</Text>
        ) : (
          <ScrollView style={styles.newsList}>
            {news.length > 0 ? (
              news.map((article, index) => (
                <TouchableOpacity 
                  key={index} 
                  onPress={() => handleNewsItemPress(article)} 
                  style={styles.newsItem}
                >
                  <Text style={styles.newsTitle}>{article.title}</Text>
                  <Text style={styles.newsDescription}>{article.description}</Text>
                </TouchableOpacity>
              ))
            ) : (
              <Text style={styles.noNewsText}>No news available at the moment.</Text>
            )}
          </ScrollView>
        )}
      </View>

      <View style={styles.safetyContainer}>
        <Text style={styles.sectionTitle}>Safety Procedures</Text>
        <Text style={styles.safetyContent}>1. Evacuate immediately if you are in danger.</Text>
        <Text style={styles.safetyContent}>2. Call emergency services if necessary.</Text>
        <Text style={styles.safetyContent}>3. Use fire extinguishers for small fires.</Text>
        <Text style={styles.safetyContent}>4. Stay calm and follow your evacuation plan.</Text>
        <Text style={styles.safetyContent}>5. Keep your family informed and safe.</Text>

        {/* Fire Department Contact */}
        <View style={styles.contactContainer}>
          <Text style={styles.contactTitle}>Emergency Contacts:</Text>
          <Text style={styles.contactInfo}>Fire Department: 911</Text>
          <Text style={styles.contactInfo}>Non-Emergency Fire Department: (555) 123-4567</Text>
          <Text style={styles.contactInfo}>Emergency Services: 911</Text>
        </View>
      </View>

      {/* Modal for full news description */}
      <Modal
        visible={modalVisible}
        animationType="slide"
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalContainer}>
          <Text style={styles.modalTitle}>{selectedArticle?.title}</Text>
          <Text style={styles.modalContent}>{selectedArticle?.content || selectedArticle?.description}</Text>
          <TouchableOpacity 
            style={styles.closeButton} 
            onPress={() => setModalVisible(false)}
          >
            <Text style={styles.closeButtonText}>Close</Text>
          </TouchableOpacity>
        </View>
      </Modal>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  newsContainer: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.7)', // Dark background for news
    borderRadius: 15,
    padding: 20,
    marginBottom: 10,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#8B0000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 4.65,
    elevation: 8,
  },
  safetyContainer: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.7)', // Dark background for safety section
    borderRadius: 15,
    padding: 20,
    marginTop: 10,
    shadowColor: '#8B0000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 4.65,
    elevation: 8,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 10,
    textAlign: 'center',
  },
  newsList: {
    marginTop: 10,
  },
  newsItem: {
    backgroundColor: '#fff',
    marginBottom: 10,
    padding: 10,
    borderRadius: 8,
  },
  newsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#000',
  },
  newsDescription: {
    fontSize: 14,
    color: '#555',
    marginTop: 5,
  },
  safetyContent: {
    fontSize: 16,
    color: '#fff',
    marginBottom: 15,
    textAlign: 'left',
  },
  errorText: {
    fontSize: 18,
    color: '#ff0000',
    textAlign: 'center',
  },
  contactContainer: {
    marginTop: 10,
    padding: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 10,
    marginBottom: 15,
  },
  contactTitle: {
    fontSize: 15,
    color: '#fff',
    fontWeight: 'bold',
    marginBottom: 5,
  },
  contactInfo: {
    fontSize: 12,
    color: '#fff',
  },
  noNewsText: {
    fontSize: 18,
    color: 'white',
    textAlign: 'center',
    marginTop: 20,
  },
  modalContainer: {
    flex: 1,
    padding: 20,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#000',
    marginBottom: 10,
  },
  modalContent: {
    fontSize: 16,
    color: '#333',
    textAlign: 'center',
  },
  closeButton: {
    marginTop: 20,
    padding: 10,
    backgroundColor: '#ff6a00',
    borderRadius: 5,
  },
  closeButtonText: {
    fontSize: 16,
    color: '#fff',
  },
});
