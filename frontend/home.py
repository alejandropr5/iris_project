import streamlit as st
import requests
import json

url = "http://0.0.0.0:8000/iris/predict"

def call_api(sepal_length, sepal_width, petal_length, petal_width):
    request_data = json.dumps([{"sepal_length": sepal_length,
                                "sepal_width": sepal_width,
                                "petal_length": petal_length,
                                "petal_width": petal_width}])
    
    headers = {'Content-Type': 'application/json'}
    response = requests.request("POST", url, headers=headers, data=request_data)
    response_json = response.json()
    predictions = response_json['predictions']
    return predictions[0]

def app():
    st.set_page_config(page_title="Streamlit App", 
                       page_icon="🎪",
                       layout="wide")
    st.title('Home')
    st.write('Welcome to the home page!!!')

    sepal_length = st.number_input('Sepal length', min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input('Sepal width', min_value=0.0, max_value=10.0, value=5.0)
    petal_length = st.number_input('Petal length', min_value=0.0, max_value=10.0, value=5.0)
    petal_width = st.number_input('Petal width', min_value=0.0, max_value=10.0, value=5.0)

    i_was_clicked = st.button("Predict")

    if i_was_clicked:
        label = call_api(sepal_length, sepal_width, petal_length, petal_width)
        st.write(f'Predicted label: {label}')
        st.balloons()

if __name__ == "__main__":
    app()