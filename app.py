import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load IoT data
data_path = 'iot_data.csv'
data = pd.read_csv(data_path)

# Preprocess the data
X = data.drop(columns=['event_type', 'timestamp', 'sensor_id'])
y = data['event_type']

# Encode categorical features
categorical_columns = ['traffic_pattern', 'incident_report', 'accident_hotspot']
label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col in categorical_columns:
    X[col] = label_encoders[col].fit_transform(X[col])

# Standardize numerical features
scaler = StandardScaler()
X[['vehicle_speed (km/h)', 'latitude', 'longitude']] = scaler.fit_transform(
    X[['vehicle_speed (km/h)', 'latitude', 'longitude']]
)

# Encode target variable
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

model_results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=y_encoder.classes_)
    model_results[model_name] = {
        'Model': model,
        'Accuracy': accuracy,
        'Report': report
    }

# Streamlit App
st.title("Smart Traffic Management: Accident Severity Prediction")

st.sidebar.header("Enter Input Features")
vehicle_speed = st.sidebar.number_input("Vehicle Speed (km/h)", min_value=0.0, max_value=200.0, step=1.0)
latitude = st.sidebar.number_input("Latitude", step=0.0001, format="%.4f")
longitude = st.sidebar.number_input("Longitude", step=0.0001, format="%.4f")
traffic_pattern = st.sidebar.selectbox("Traffic Pattern", options=label_encoders['traffic_pattern'].classes_)
incident_report = st.sidebar.selectbox("Incident Report", options=label_encoders['incident_report'].classes_)
accident_hotspot = st.sidebar.selectbox("Accident Hotspot", options=label_encoders['accident_hotspot'].classes_)

# Model selection
st.sidebar.subheader("Select Models for Prediction")
model_selection = st.sidebar.multiselect(
    "Choose models or select 'ALL'",
    options=["ALL"] + list(models.keys()),
    default="ALL"
)

if st.sidebar.button("Predict Severity"):
    # Prepare user input
    user_input = pd.DataFrame({
        'vehicle_speed (km/h)': [vehicle_speed],
        'latitude': [latitude],
        'longitude': [longitude],
        'traffic_pattern': [label_encoders['traffic_pattern'].transform([traffic_pattern])[0]],
        'incident_report': [label_encoders['incident_report'].transform([incident_report])[0]],
        'accident_hotspot': [label_encoders['accident_hotspot'].transform([accident_hotspot])[0]]
    })

    # Scale numerical features
    numerical_features = ['vehicle_speed (km/h)', 'latitude', 'longitude']
    scaled_numerical = scaler.transform(user_input[numerical_features])

    # Combine scaled numerical and categorical features
    processed_input = np.concatenate(
        [scaled_numerical, user_input[['traffic_pattern', 'incident_report', 'accident_hotspot']].values], axis=1
    )

    # Validate feature shape
    assert processed_input.shape == (1, X_train.shape[1]), (
        f"Expected shape (1, {X_train.shape[1]}), but got {processed_input.shape}"
    )

    # Select models
    selected_models = models.keys() if "ALL" in model_selection else model_selection

    # Predict and display results
    st.write("### Prediction Results")
    for model_name in selected_models:
        model = model_results[model_name]['Model']
        prediction = model.predict(processed_input)
        severity = y_encoder.inverse_transform(prediction)[0]
        st.write(f"**{model_name}:** {severity}")

    # Model Comparison
    st.subheader('Model Comparison')
    comparison = {}
    for model_name in selected_models:
        y_pred = models[model_name].predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        comparison[model_name] = accuracy

    comparison_df = pd.DataFrame(list(comparison.items()), columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
    st.dataframe(comparison_df)
