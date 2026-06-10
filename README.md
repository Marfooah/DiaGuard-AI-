# DiaGuard AI

### Intelligent Diabetes Risk Screening with Artificial Neural Networks

DiaGuard AI is a modern machine learning application that predicts diabetes risk using clinical health indicators and an Artificial Neural Network (ANN). Built on the Pima Indians Diabetes Dataset, the system transforms patient measurements into an interpretable risk assessment, helping users understand both their predicted risk level and the factors contributing to it.

The project combines machine learning, data preprocessing, explainable AI techniques, and a premium interactive Streamlit interface to deliver a professional healthcare analytics experience.

---

## Demo

https://diaguard.streamlit.app

---

## Screenshots

### Dashboard

<img width="2950" height="1736" alt="image" src="https://github.com/user-attachments/assets/76782797-67b5-48ce-809c-141d00ed8b52" />
<img width="2950" height="1620" alt="image" src="https://github.com/user-attachments/assets/96ac4dff-c857-4451-84bd-885a7de40b39" />
<img width="2950" height="982" alt="image" src="https://github.com/user-attachments/assets/c86b571a-a5de-418e-aadc-957436333f61" />

### Risk Analysis

<img width="2950" height="1732" alt="image" src="https://github.com/user-attachments/assets/9adca019-6b89-4960-b649-14048c9a5d75" />
<img width="2950" height="1730" alt="image" src="https://github.com/user-attachments/assets/9a16eaea-9faa-4e0b-88fd-b4f91eed8590" />
<img width="2950" height="998" alt="image" src="https://github.com/user-attachments/assets/088129ba-b3d2-4ddf-bedd-0556b873dfd4" />

### Model Performance

<img width="2950" height="1726" alt="image" src="https://github.com/user-attachments/assets/5cc4f94e-8ee1-46ca-96b8-a3aa895e5359" />
<img width="2950" height="644" alt="image" src="https://github.com/user-attachments/assets/276bcd21-2209-4db0-898a-e4e8ce1a0f54" />

---

## Features

* AI-powered diabetes risk prediction
* Artificial Neural Network (ANN) classifier
* Interactive Streamlit dashboard
* Real-time risk scoring
* Feature impact analysis
* Clinical health insights and recommendations
* Population median comparison charts
* Risk probability visualization
* Multiple patient profile presets
* Automatic preprocessing pipeline
* TensorFlow/Keras backend with Scikit-Learn fallback
* Modern premium UI with Plotly visualizations

---

## Problem Statement

Diabetes is one of the most common chronic diseases worldwide. Early identification of individuals at elevated risk can support timely medical consultation and lifestyle interventions.

DiaGuard AI leverages machine learning to analyze patient health metrics and estimate diabetes risk through a data-driven screening process.

**Note:** This application is designed for educational and screening purposes only and should not be used as a substitute for professional medical diagnosis.

---

## Dataset

**Pima Indians Diabetes Dataset**

The model was trained using the widely used Pima Indians Diabetes Dataset containing clinical records of female patients of Pima Indian heritage.

### Dataset Characteristics

| Attribute                  | Description                  |
| -------------------------- | ---------------------------- |
| Pregnancies                | Number of pregnancies        |
| Glucose                    | Plasma glucose concentration |
| Blood Pressure             | Diastolic blood pressure     |
| Skin Thickness             | Triceps skin fold thickness  |
| Insulin                    | 2-hour serum insulin         |
| BMI                        | Body Mass Index              |
| Diabetes Pedigree Function | Genetic predisposition score |
| Age                        | Patient age                  |
| Outcome                    | Diabetes diagnosis           |

Dataset Size: **768 patient records**

---

## Model Architecture

The primary model uses a deep Artificial Neural Network implemented with TensorFlow/Keras.

### Network Structure

```text
Input Layer (8 Features)
        ↓
Dense (64, ReLU)
        ↓
Batch Normalization
        ↓
Dropout (15%)
        ↓
Dense (32, ReLU)
        ↓
Batch Normalization
        ↓
Dropout (10%)
        ↓
Dense (16, ReLU)
        ↓
Dense (1, Sigmoid)
```

### Training Enhancements

* Early Stopping
* Learning Rate Reduction
* Class Weight Balancing
* Standardization using StandardScaler
* Missing Value Imputation using Median Strategy

---

## Data Preprocessing Pipeline

The following preprocessing steps are applied before prediction:

1. Replace invalid zero values with missing values
2. Median imputation
3. Feature standardization
4. ANN inference
5. Probability-based classification

### Columns Treated for Missing Values

```python
[
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]
```

---

## Explainable AI Features

DiaGuard AI provides more than a binary prediction.

### Risk Probability

The model outputs a probability score between 0% and 100%.

### Feature Impact Analysis

The application estimates how strongly each clinical measurement contributes to the final risk score.

### Health Recommendations

Personalized guidance is generated based on patient metrics including:

* Glucose levels
* BMI
* Blood pressure
* Family history
* Age-related risk factors

---

## Technology Stack

### Machine Learning

* TensorFlow / Keras
* Scikit-Learn
* NumPy
* Pandas
* Joblib

### Frontend & Visualization

* Streamlit
* Plotly

### Development

* Python

---

## Project Structure

```text
DiaGuard-AI/
│
├── app.py
├── predict.py
├── train_model.py
├── diabetes.csv
├── diabetes_mip_model.joblib
├── diabetes_preprocessing.joblib
├── model_metrics.joblib
├── requirements.txt
├── Diabetes_Classifier.ipynb
└── README.md
    ├── dashboard.png
    ├── risk-analysis.png
    └── model-performance.png
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/DiaGuard-AI.git

cd DiaGuard-AI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train_model.py
```

### Launch Application

```bash
streamlit run app.py
```

---

## Model Output

The system provides:

* Diabetes Risk Percentage
* Positive / Negative Screening Result
* Risk Category
* Feature Contribution Analysis
* Population Comparison
* Health Insights

### Risk Categories

| Probability | Category       |
| ----------- | -------------- |
| < 35%       | Low Risk       |
| 35% – 55%   | Moderate Risk  |
| 55% – 75%   | High Risk      |
| > 75%       | Very High Risk |

---

## Future Improvements

* SHAP explainability integration
* Model monitoring dashboard
* Electronic Health Record integration
* Multi-disease screening support
* Advanced deep learning architectures
* Clinical report generation
* User authentication and patient history tracking

---

## Disclaimer

DiaGuard AI is intended solely for educational, research, and demonstration purposes.

The predictions generated by this application do not constitute medical advice, diagnosis, or treatment recommendations. Always consult a qualified healthcare professional regarding any health concerns.

---

## Author

**Ayesha Tariq**

AI & Machine Learning Enthusiast • Builder of AI-powered solutions

## License

This project is licensed under the MIT License.

If you found this project interesting, consider giving the repository a ⭐.
