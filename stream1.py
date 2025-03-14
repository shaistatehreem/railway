import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load data
def load_data():
    return pd.read_csv(r"D:\\Train_Accident_prediction\\data\\train_collisions_india.csv")

df = load_data()

# Sidebar
st.sidebar.title("Railway Accident Prediction")
option = st.sidebar.selectbox("Select Analysis", (
    "Data Overview",
    "Visualizations",
    "Model Training",
    "Risk Prediction"
))

if option == "Data Overview":
    st.title("Data Overview")
    st.write(df.head())
    st.write("Missing values:", df.isnull().sum())
    st.write("Data types:", df.dtypes)

if option == "Visualizations":
    st.title("Data Visualizations")

    # Yearly Trend
    st.subheader("Yearly Trend of Railway Accidents")
    fig, ax = plt.subplots()
    df['Year'].value_counts().sort_index().plot(kind='line', marker='o', color='b', ax=ax)
    plt.xlabel("Year")
    plt.ylabel("Number of Accidents")
    plt.grid(True)
    st.pyplot(fig)

    # Causes of Accidents
    st.subheader("Most Common Causes of Railway Accidents")
    fig, ax = plt.subplots()
    sns.countplot(y=df['Cause'], order=df['Cause'].value_counts().index, palette="viridis", ax=ax)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    st.pyplot(fig)

if option == "Model Training":
    st.title("Model Training")
    label_encoder = LabelEncoder()
    df['Cause'] = label_encoder.fit_transform(df['Cause'])
    df['Weather'] = label_encoder.fit_transform(df['Weather'])
    scaler = StandardScaler()
    df[['Train Speed', 'Casualties']] = scaler.fit_transform(df[['Train Speed', 'Casualties']])

    X = df[['Location', 'Weather', 'Train Speed', 'Maintenance Status', 'Signal Error']]
    y = df['Risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)

    mse = mean_squared_error(y_test, rf_regressor.predict(X_test))
    r2 = r2_score(y_test, rf_regressor.predict(X_test))

    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")

    joblib.dump(rf_regressor, 'random_forest_model.pkl')

if option == "Risk Prediction":
    st.title("Risk Prediction")

    location = st.number_input("Location Code", min_value=0)
    weather = st.selectbox("Weather", options=[0, 1, 2])
    speed = st.number_input("Train Speed", min_value=0.0)
    maintenance = st.selectbox("Maintenance Status", options=[0, 1])
    signal_error = st.selectbox("Signal Error", options=[0, 1])

    if st.button("Predict Risk"):
        model = joblib.load('random_forest_model.pkl')
        input_df = pd.DataFrame({
            'Location': [location],
            'Weather': [weather],
            'Train Speed': [speed],
            'Maintenance Status': [maintenance],
            'Signal Error': [signal_error]
        })

        prediction = model.predict(input_df)[0]
        prediction = max(-1, min(1, prediction))

        st.success(f"Predicted Risk: {prediction * 100:.2f}%")
