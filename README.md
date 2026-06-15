# Sleep Disruption and Attention Fragmentation

This project investigates the relationship between sleep patterns, stress, digital wellbeing, and their impact on attention fragmentation and sleep quality. It uses machine learning models to predict the level of attention fragmentation based on various lifestyle factors.

## Dataset
The dataset includes several features such as:
- **Daily Screen Time (minutes)**
- **Phone Usage Before Sleep (minutes)**
- **Sleep Duration (minutes)**
- **Sleep Efficiency Score**
- **Stress & Fatigue Index**
- **Digital Wellbeing Score**
- **Occupation Categories**

## Modeling
We experimented with various models for both Classification (predicting Attention Fragmentation level: Low, Moderate, High) and Regression (predicting Sleep Quality Score).

### Classification Models Used:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

**Best Model**: Random Forest Classifier (Accuracy ~93.4%) with SMOTE for handling imbalanced classes.

## Project Structure
```
├── data/                      # Dataset (ignored in git)
├── notebooks/                 # Jupyter notebooks for EDA and modeling
├── models/                    # Exported joblib models and scalers
├── src/                       # Source scripts for training models
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## How to Run the Web App Locally

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Sleep-Disruption-and-Attention-Fragmentation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Deployment
The application is deployed using Streamlit Community Cloud. You can access it here: [Insert Deployed URL]
