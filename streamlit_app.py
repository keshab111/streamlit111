import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Title
st.title("Solar Power Generation Prediction")

# Load data
@st.cache
def load_data():
    return pd.read_csv('solarpowergeneration.csv')

data = load_data()

# Display data
if st.checkbox("Show raw data"):
    st.write(data)

# EDA: Display a histogram
st.subheader("Distribution of Power Generated")
fig, ax = plt.subplots()
ax.hist(data['power_generated'], bins=30)
st.pyplot(fig)

# Feature Selection
st.subheader("Select Features")
features = st.multiselect("Features", data.columns[:-1], default=list(data.columns[:-1]))

# Train/Test Split
X = data[features]
y = data['power_generated']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display Results
st.subheader("Model Performance")
st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
st.write("R^2 Score:", r2_score(y_test, y_pred))
