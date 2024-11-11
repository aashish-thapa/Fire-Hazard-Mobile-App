# My CNN API

A FastAPI application serving predictions from a Convolutional Neural Network (CNN) model.

## Features

- Upload images and receive predictions.
- Secure API endpoints with API keys.
- Deployable on AWS Elastic Beanstalk or Heroku.

## Setup

### Local Development

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/my_cnn_api.git
    cd my_cnn_api
    ```

2. **Create and Activate Virtual Environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r app/requirements.txt
    ```

4. **Run the Application:**

    ```bash
    uvicorn app.main:app --reload
    ```

5. **Access API Documentation:**

    Visit `http://localhost:8000/docs` in your browser.

## Deployment

### AWS Elastic Beanstalk

1. **Initialize Elastic Beanstalk:**

    ```bash
    eb init -p docker my-cnn-api
    ```

2. **Create Environment and Deploy:**

    ```bash
    eb create my-cnn-api-env
    eb deploy
    eb open
    ```

### Heroku

1. **Login to Heroku:**

    ```bash
    heroku login
    ```

2. **Create and Deploy:**

    ```bash
    heroku create your-app-name
    git push heroku main
    heroku open
    ```

## Usage

### Prediction Endpoint

- **URL:** `/predict`
- **Method:** `POST`
- **Headers:**
  - `API_KEY: your_secure_api_key`
- **Body:**
  - Form-data with a key `file` containing the image.

### Example with `curl`:

```bash
curl -X POST "http://your-deployment-url/predict" \
  -H "API_KEY: your_secure_api_key" \
  -F "file=@path_to_your_image.jpg"

