# Fire Prediction App

This Fire Prediction App is designed to assist in monitoring and predicting fire spread based on user location and active fire data. Built with React Native and integrated with real-time satellite data, this app provides insights into possible fire-evolving locations and push notifications to alert users in high-risk areas.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [APIs](#apis)
- [File Structure](#file-structure)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Overview
The Fire Prediction App uses a combination of satellite data, machine learning, and geolocation to predict fire spread and notify users in affected regions. By analyzing active fire data and using generative AI, the app shows potential fire spread areas on a map, providing users with an interface to monitor fire hazards nearby.

## Features
- **Map with User and Fire Locations**: Shows the user's location and marks active fires nearby.
- **Fire Spread Prediction**: Uses a machine learning model to predict potential fire spread, visualized as polygons on the map.
- **Push Notifications**: Alerts users when they are in high-risk zones.
- **Satellite Data Integration**: Displays satellite map views of areas with active fires.
- **Safety Measures and News Feed**: Access the latest fire hazard news and recommended safety precautions for nearby areas.

## Technologies Used
- **React Native**: For cross-platform mobile application development.
- **Expo Router**: To manage screen navigation and routing.
- **Firebase Cloud Messaging (FCM)**: For sending push notifications.
- **MapView**: For rendering maps and displaying fire spread predictions.
- **TensorFlow**: Used for the fire spread prediction model.
- **REST APIs**: To fetch fire data and send prediction requests.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/aashish-thapa/fire-prediction-app.git
    cd fire-prediction-app
    ```

2. **Install Dependencies**:
    ```bash
    npm install
    ```

3. **Set Up Firebase**:
   - Create a Firebase project and enable Firebase Cloud Messaging.
   - Download the `google-services.json` file and add it to the project directory.

4. **Set Environment Variables**:
   - In a `.env` file, add the following credentials:
     ```
     FIRE_API_URL=https://data.api.xweather.com/fires/closest
     CLIENT_ID=gzSR9NjGiYdzGzItC4BpE
     CLIENT_SECRET=UdZ2WV8o7vb55UG4Jd4CIExUHfXm8CyMFaWu454F
     ```

5. **Run the Project**:
    ```bash
    npm start
    ```

6. **Start the Emulator**:
   - Open the Expo app on your mobile device or use an Android/iOS emulator to run the application.

## Usage

1. **Get Fire Data**:
   - When the app loads, it will request the user’s location to display their position on the map.
   - The app fetches nearby active fires and displays their locations as markers on the map.

2. **Fetch Fire Spread Prediction**:
   - Tap on "Fetch Prediction Data" to send a request to the fire prediction API. The app will visualize the potential spread areas around each fire location with varying intensity zones.

3. **Receive Notifications**:
   - If you enter a high-risk area, you will receive a push notification.

4. **Safety News Feed**:
   - Access the latest fire hazard news and safety precautions from the app.

## APIs

### XWeather Fire Data API
- **Endpoint**: `https://data.api.xweather.com/fires/closest`
- **Parameters**:
  - `p`: The location postal code (for example, `39406`).
  - `client_id`: Your XWeather API client ID.
  - `client_secret`: Your XWeather API client secret.

### Prediction API
- **Endpoint**: `http://{for ip contact the repo owner}/predict/`
- **Method**: `POST`
- **Request Data**:
    ```json
    {
      "data": [
        [
          [0.0, 180.0, 5.0, 275.0, 290.0, 0.01, 1.5, 0.0, 0.6, 2.0, 40.0, -0.1],
          ...
        ]
      ]
    }
    ```
- **Response**: Predicted fire spread data.

## File Structure

fire-prediction-app/ │ ├── assets/ │ └── images/ │ └── fire.png # Fire icon for map markers ├── components/ │ └── Prediction.tsx # Main prediction component ├── screens/ │ ├── HomeScreen.tsx # Home screen with map and data │ └── PredictionScreen.tsx # Screen displaying prediction map and fire zones ├── App.tsx # Main application file ├── README.md # Project README file ├── package.json # Project dependencies └── .env # Environment variables


## Future Improvements
- **Enhanced Prediction Model**: Improve accuracy and efficiency with additional features.
- **Weather Data Integration**: Factor in real-time weather conditions to adjust predictions.
- **Real-Time Location Tracking**: Continuously track user location to update notifications in real-time.
- **Offline Support**: Allow offline access to previous fire data and alerts.
- **User Preferences**: Enable users to customize notification and data settings.

## License
This project is licensed under the MIT License.

## Contact
aashishthapa520@gmail.com
