import streamlit as st
import numpy as np
import pickle


from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score


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

models = {
    'lr':LinearRegression(),
    'lss':Lasso(),
    'rg':Ridge(),
    'Knr':KNeighborsRegressor(),
    'dtr':DecisionTreeRegressor()
}

dtr = DecisionTreeRegressor()
dtr.fit(x_train_dummy,y_train)
dtr.predict(x_test_dummy)
for name, mod in models.items():
    mod.fit(x_train_dummy,y_train)
    y_pred = mod.predict(x_test_dummy)
    
    print(f"{name} MSE : {mean_squared_error(y_test,y_pred)} Score {r2_score(y_test,y_pred)}")

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
x_train_dummy = preprocesser.fit_transform(x_train)
x_test_dummy = preprocesser.transform(x_test)
    if st.button('Predict'):
        from sklearn.preprocessing import OneHotEncoder,StandardScaler
        from sklearn.compose import ColumnTransformer
        ohe = OneHotEncoder(drop = 'first')
        scaler = StandardScaler()
        preprocesser = ColumnTransformer(
        transformers = [
            ('onehotencoder',ohe,[0,1]),
            ('standrization',scaler,[2,3,4,5])
        ],
        remainder = 'passthrough'
        )
        
        predicted_value = predict_yield(dtr, preprocesser, area, item, year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp)

        st.markdown('## Predicted Yield Productions:')
        st.write(predicted_value)
