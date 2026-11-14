# 🌳 Random Forest Classification – Car Evaluation Dataset 🚗

## 📌 Project Description
This project implements a Random Forest Classifier using Python and Machine Learning to predict car acceptability based on various features such as buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, and safety level.

The project also compares model performance with all features and after removing the `doors` feature.

## 🎯 Objectives
- Understand Random Forest Classification.
- Perform data preprocessing and categorical encoding.
- Train and test a Random Forest model.
- Evaluate model performance.
- Visualize feature importance.
- Compare predictions with and without the doors feature.

## 📂 Dataset Information

**Dataset:** Car Evaluation Dataset

| Feature | Description |
|---|---|
| buying | Buying price |
| maint | Maintenance cost |
| doors | Number of doors |
| persons | Passenger capacity |
| lug_boot | Luggage boot size |
| safety | Safety level |
| class | Car evaluation category (Target) |

### 🎯 Target Classes
- `unacc` – Unacceptable
- `acc` – Acceptable
- `good` – Good
- `vgood` – Very Good

## 🤖 Machine Learning Algorithm

**Random Forest Classifier**

- Number of estimators: 100
- Random state: 0
- Train-test split: 67% Training / 33% Testing

### 🔍 Model Experiments
1. Random Forest with all six features.
2. Random Forest after removing the `doors` feature.

## 🛠️ Technologies & Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## ⚙️ Project Workflow
1. Import required libraries.
2. Load the Car Evaluation dataset.
3. Perform data inspection.
4. Check missing values.
5. Encode categorical features using OrdinalEncoder.
6. Split data into training and testing sets.
7. Train the Random Forest Classifier.
8. Predict car evaluation categories.
9. Calculate model accuracy.
10. Visualize feature importance.
11. Generate confusion matrix and classification report.
12. Compare model performance after removing the doors feature.

## 📊 Model Evaluation
The model is evaluated using:
- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-Score
- Feature Importance

## 📈 Visualization
The project generates a feature importance graph to identify which features contribute most to car evaluation predictions.

**Output file:** `feature_importance.png`

## 📁 Project Structure

```text
Random-Forest-Car-Evaluation/
│
├── car_evaluation.csv
├── random_forest.py
├── feature_importance.png
└── README.md
```

## ▶️ How to Run the Project

### 1. Clone the repository
```bash
git clone YOUR_REPOSITORY_URL
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the Python file
```bash
python random_forest.py
```

## 🎓 Learning Outcomes
- Understanding Random Forest Classification.
- Working with categorical datasets.
- Applying feature encoding.
- Training and evaluating ML models.
- Understanding feature importance.
- Comparing machine learning experiments.

## 👨‍💻 Author
**Shaik Nawaz Shareef**

Machine Learning | Artificial Intelligence | Python

---
⭐ If you find this project useful, consider giving it a star!
