# **Illness Prediction System**

## **Overview**

The **Illness Prediction System** is a machine learning-powered web application that helps users identify potential illnesses based on their symptoms and recommends a specialist doctor for further consultation. The goal of this project is to assist users in making informed healthcare decisions efficiently and effectively.

## **Features**

- **Symptom-Based Predictions**: Users input their symptoms, and the system predicts potential illnesses using a trained machine learning model.
- **Doctor Recommendations**: The system suggests specialist doctors based on the predicted illness.
- **User-Friendly Web Interface**: A responsive and intuitive interface for ease of use.
- **Secure and Scalable**: Designed for scalability and security, making it suitable for deployment in production environments.

## **Project Structure**

```
illness_prediction/
├── app/                      # Web application backend/frontend
│   ├── static/               # Static files (CSS, JS, images)
│   ├── templates/            # HTML templates for the web interface
│   ├── app.py                # Main application script
│   ├── routes.py             # Route definitions
│   └── requirements.txt      # App-specific dependencies
│
├── data/                     # Data storage and preprocessing
│   ├── raw/                  # Raw data files
│   ├── processed/            # Processed/cleaned data
│   ├── external/             # External data sources (optional)
│   └── data_preprocessing.py # Scripts for data cleaning/preprocessing
│
├── models/                   # Machine learning models
│   ├── trained_models/       # Saved trained models
│   ├── training/             # Training scripts
│   │   ├── train.py          # Main training script
│   │   └── utils.py          # Helper functions for training
│   └── illness_model.py      # Code for model definition and inference
│
├── notebooks/                # Jupyter notebooks for experimentation
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   ├── model_dev.ipynb       # Model development
│   └── preprocessing.ipynb   # Data preprocessing
│
├── tests/                    # Unit and integration tests
│   ├── test_data.py          # Tests for data preprocessing
│   ├── test_model.py         # Tests for model predictions
│   └── test_app.py           # Tests for API endpoints and app functionality
│
├── config/                   # Configuration files
│   ├── config.yaml           # General configuration
│   └── logging_config.py     # Logging configuration
│
├── logs/                     # Logs for debugging
├── docker/                   # Docker configuration
│   ├── Dockerfile            # Dockerfile for containerization
│   └── docker-compose.yml    # Multi-service setup
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .gitignore                # Files to exclude from version control
└── setup.py                  # Packaging script (optional)

```

## **Installation**

### Step 1: Clone the Repository

```cmd
git clone https://github.com/SenhadjiMSaid/illness_prediction.git
cd illness_prediction
```

### **Step 2: Install Dependencies**

Create a `conda` environment using the provided `environment.yml` file:

```cmd
conda env create -f environment.yml
```

Activate the environment:

```cmd
conda activate illness_prediction
```

If you need to add new packages in the future, update the environment.yml file and use:

```cmd
conda env update --file environment.yml --prune
```

## **Usage**

### **Running the Application**

1. Navigate to the `app/` directory.
2. Start the application:

```cmd
 python app.py
```

3. Open a browser and go to http://127.0.0.1:5000 or <a>http://[your-computer-ipv4-address]:5000</a>

### **Interacting with the System**

- **Step 1**: Enter your symptoms in the input form on the web interface.
- **Step 2**: Click _"Predict"_ to get a list of potential illnesses.
- **Step 3**: View recommended specialists for consultation.

## **Model Training**

To train the model from scratch:

1. Navigate to the `models/training/` directory.
2. Run the training script:

```cmd
python train.py
```

3. he trained model will be saved in the `models/trained_models/` folder.

## **Testing**

Run the tests to ensure the application works as expected:

```cmd
pytest tests/
```

## **Deployment**

This application can be deployed using Docker:

1. Build the Docker image:
2. Run the container:

## **Technologies Used**

- Frontend: HTML, CSS, JavaScript
- Backend: Python (Flask)
- Machine Learning: Scikit-learn, TensorFlow/Keras, or PyTorch
- Data Processing: Pandas, NumPy, ydata_profiling
- Testing: Pytest
- Containerization: Docker
