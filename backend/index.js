const express = require('express');
const db = require('./db');
const app = express();

app.use(express.json());

// Endpoint to add a new user
app.post('/addUser', (req, res) => {
  const { user_name, user_email, geolocation } = req.body;
  const sql = `INSERT INTO users (user_name, user_email, geolocation) VALUES (?, ?, ?)`;

  db.query(sql, [user_name, user_email, geolocation], (err, result) => {
    if (err) {
      return res.status(500).send({ error: err.message });
    }
    res.status(200).send({ message: 'User added successfully', id: result.insertId });
  });
});

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
