import streamlit as st 
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(
    page_title="Iris Classification", 
    initial_sidebar_state="expanded",
    page_icon=":sunflower",
    layout="centered",
)

# Main body
st.header("Iris Classification")
st.write("This app predicts the Iris flower type")

# Sidebar
st.sidebar.header("User Input Parameters")

# Function for user input features
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 3.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.9)
    
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
test_data = user_input_features()
st.subheader('User Input Parameters')
st.dataframe(test_data, hide_index=True)

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Make predictions
prediction = clf.predict(test_data)
prediction_proba = clf.predict_proba(test_data)

# Display the results
st.subheader('Class Labels and Their Corresponding Index Number')
class_labels_df = pd.DataFrame({
    'Index Number': range(len(iris.target_names)),
    'Class Label': iris.target_names
})
st.dataframe(class_labels_df)

st.subheader('Prediction')
st.write(f"Predicted class index: {prediction[0]}")
st.write(f"Predicted class label: {iris.target_names[prediction][0]}")

st.subheader('Prediction Probability')
st.write(prediction_proba)