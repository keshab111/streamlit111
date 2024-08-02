import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression


      class LogisticRegressor:
          def __init__(self, df):
           self.df = df
            self.model = LogisticRegression()

         def preprocess_data(self):
          # Assuming 'target' is the column to be predicted and others are features
         X = self.df.drop(columns=['target'])
         y = self.df['target']
         return train_test_split(X, y, test_size=0.2, random_state=42)

         def train(self, X_train, y_train):
         self.model.fit(X_train, y_train)
         st.write("Logistic regression model trained.")

        def evaluate(self, X_test, y_test):
         y_pred = self.model.predict(X_test)
         accuracy = accuracy_score(y_test, y_pred)
         conf_matrix = confusion_matrix(y_test, y_pred)
         class_report = classification_report(y_test, y_pred)
    
            st.write("### Model Performance")
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write("Confusion Matrix:")
            st.write(conf_matrix)
            st.write("Classification Report:")
            st.write(class_report)
