import streamlit as st
from DataProcessor.DataLoader import DataLoader
from DataProcessor.DataEvaluator import DataEvaluator
from DataProcessor.GraphicGenerator import GraphicGenerator
from DataProcessor.LogisticRegressor import LogisticRegressor

# To run the APP: streamlit run streamlit_app.py
if __name__ == '__main__':
    st.header('Logistic Regression')
    st.markdown('*Author: Keshab Chandra Padhi*')

    # Data Loader
    st.header('Data loader')
    dataLoader = DataLoader()
    dataLoader.check_labels()
    dataLoader.check_separator()
    file = dataLoader.load_file()

    if file is not None:
        df = dataLoader.load_data(file)

        # Data evaluation
        st.header('Data evaluation')
        st.write('Non-numeric columns and rows with missing values have been dropped.')
        dataEvaluator = DataEvaluator(df)
        dataEvaluator.show_head()
        dataEvaluator.show_dimensions()
        dataEvaluator.show_columns()

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
