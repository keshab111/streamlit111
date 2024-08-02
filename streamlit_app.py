import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
from DataProcessor.DataLoader import DataLoader
from DataProcessor.DataEvaluator import DataEvaluator
from DataProcessor.GraphicGenerator import GraphicGenerator
from DataProcessor.LogisticRegressor import LogisticRegressor 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# To run the APP: streamlit run streamlit_app.py

if __name__ == '__main__':
    st.header('Logistic Regression')
    st.markdown('*Author:  KESHAB CHANDRA PADHI*')

    # Data Loader
    st.header('Data loader')
    dataLoader = DataLoader() 
    dataLoader.check_labels()
    dataLoader.check_separator()
    file = dataLoader.load_file()

# Title
st.title("Solar Power Generation Prediction")


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


        # Graphic Plots
        st.header('Graphic Plots')
        plotGenerator = GraphicGenerator(df)

        checked_pairplot = st.checkbox('PairPlot')
        checked_scatterPlot = st.checkbox('ScatterPlot')
        checked_correlationPlot = st.checkbox('Correlation')
        checked_logisticRegPlot = st.checkbox('LogisticRegPlot')

        if checked_pairplot:
            plotGenerator.pairplot()
            st.markdown('<hr/>', unsafe_allow_html=True)

        if checked_scatterPlot:
            plotGenerator.scatterplot()
            st.markdown('<hr/>', unsafe_allow_html=True)

        if checked_correlationPlot:
            plotGenerator.correlationPlot()
            st.markdown('<hr/>', unsafe_allow_html=True)

        if checked_logisticRegPlot:
            plotGenerator.logisticRegressionPlot()
            st.markdown('<hr/>', unsafe_allow_html=True)

        # Logistic Regression
        st.header('Logistic Regression')
        regressor = LogisticRegressor(df)
        regressor.logistic()
        st.markdown('<hr/>', unsafe_allow_html=True)
