import streamlit as st
import numpy as np
import pickle



def load_models():
    try:
        # Loading models
        dtr = pickle.load(open('dtr.pkl', 'rb'))
        preprocesser = pickle.load(open('preprocesser.pkl', 'rb'))
        return dtr, preprocesser
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def predict_yield(dtr, preprocesser, area, item, year, rainfall, pesticides, temperature):
    features = np.array([[area, item, year, rainfall, pesticides, temperature]])
    transformed_features = preprocesser.transform(features) 
    predicted_value = dtr.predict(transformed_features).reshape(1, -1)
    return predicted_value

st.title('Crop Yield Prediction Per Country')

dtr, preprocesser = load_models()

if dtr is not None and preprocesser is not None:
    st.markdown('## Crop Features Here')

    year = st.number_input('Year')
    average_rain_fall_mm_per_year = st.number_input('Average Rainfall (mm per year)')
    pesticides_tonnes = st.number_input('Pesticides (tonnes)')
    avg_temp = st.number_input('Average Temperature')

    area_options = ["Albania", "Algeria", "Angola","Argentina","Bahamas","Bahrain","Bangladesh","Belarus","Canada","Egypt","India","Nepal","New Zealand","Pakistan"]

    area = st.selectbox('Area', area_options)

    item_option = ["Maize","Potatoes","Wheat","Rice, paddy","Soybeans","Sweet potatoes","Cassava","Yams","Plantains and others"] 
    item = st.selectbox('Item', item_option)

    if st.button('Predict'):
        predicted_value = predict_yield(dtr, preprocesser, area, item, year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp)

        st.markdown('## Predicted Yield Productions:')
        st.write(predicted_value)
