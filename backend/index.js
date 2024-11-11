const express = require('express');
const db = require('./db');
const app = express();
const cors = require('cors');
app.use(cors());
app.use(express.json());

app.post('/storeUserData', (req, res) => {
  const { userId, email, latitude, longitude } = req.body;

  // Check if user already exists
  const checkQuery = 'SELECT * FROM users WHERE email = ?';
  db.query(checkQuery, [email], (err, result) => {
    if (err) {
      res.status(500).json({ success: false, message: 'Database query error' });
      return;
    }
    if (result.length > 0) {
      res.status(400).json({ success: false, message: 'User already exists' });
      return;
    }

    // Insert new user data
    const query = 'INSERT INTO users (userId, email, latitude, longitude) VALUES (?, ?, ?, ?)';
    db.query(query, [userId, email, latitude, longitude], (err, result) => {
      if (err) {
        res.status(500).json({ success: false, message: 'Database insertion error' });
        return;
      }
      res.status(200).json({ success: true, message: 'User data stored successfully' });
    });
  });
});

app.listen(3000, () => {
  console.log('Server is running on port 3000');
});
