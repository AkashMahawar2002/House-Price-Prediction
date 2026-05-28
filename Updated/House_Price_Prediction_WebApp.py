# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:57:51 2024

@author: prachet
"""

import json
import pickle
import streamlit as st
import pandas as pd

#loading. the saved model
all_features = None
scalers = None
best_features_lr = []
best_features_rfr = []
best_features_xgb = []
loaded_model_lr = None
loaded_model_rfr = None
loaded_model_xgb = None

def _safe_load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning(f"Missing file: {path}")
        return None
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None

def _safe_load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"Missing file: {path}")
        return []
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return []

all_features = _safe_load_pickle("Updated/columns.pkl")
scalers = _safe_load_pickle("Updated/scaler.pkl")
best_features_lr = _safe_load_json("Updated/best_features_lr.json")
best_features_rfr = _safe_load_json("Updated/best_features_rfr.json")
best_features_xgb = _safe_load_json("Updated/best_features_xgb.json")
loaded_model_lr = _safe_load_pickle("Updated/house_price_prediction_trained_lr_model.sav")
loaded_model_rfr = _safe_load_pickle("Updated/house_price_prediction_trained_rfr_model.sav")
loaded_model_xgb = _safe_load_pickle("Updated/house_price_prediction_trained_xgb_model.sav")


# converting the category columns to string data type
#cat_cols = ['hasYard', 'hasPool', 'cityPartRange', 'numPrevOwners', 'made', 'isNewBuilt','hasStormProtector', 'hasStorageRoom','hasGuestRoom']

#creating a function for prediction

def house_price_prediction(input_data):

    # build dataframe from inputs and scale
    if all_features is None or scalers is None:
        st.error("Model metadata (features/scaler) not available.")
        return None, None, None

    df = pd.DataFrame([input_data], columns=all_features)

    try:
        scaled_arr = scalers.transform(df[all_features])
    except Exception as e:
        st.error(f"Error applying scaler: {e}")
        return None, None, None

    scaled_df = pd.DataFrame(scaled_arr, columns=all_features)

    # helper to get expected feature names from a model
    def _model_expected_features(model, best_features):
        # prefer model's stored feature names if available
        if model is None:
            return best_features if best_features else None
        try:
            if hasattr(model, 'feature_names_in_'):
                return list(model.feature_names_in_)
        except Exception:
            pass
        # xgboost Booster
        try:
            booster = model.get_booster()
            if hasattr(booster, 'feature_names') and booster.feature_names is not None:
                return list(booster.feature_names)
        except Exception:
            pass
        # fall back to provided best_features
        return best_features if best_features else None

    def _prepare_input_for_model(model, best_features):
        expected = _model_expected_features(model, best_features)
        if expected is None:
            return scaled_df
        # reindex to expected columns; fill absent columns with zeros
        missing = [c for c in expected if c not in scaled_df.columns]
        if missing:
            st.warning(f"Filling missing features for prediction with zeros: {missing}")
        return scaled_df.reindex(columns=expected, fill_value=0)

    df_best_features_lr = _prepare_input_for_model(loaded_model_lr, best_features_lr)
    df_best_features_rfr = _prepare_input_for_model(loaded_model_rfr, best_features_rfr)
    df_best_features_xgb = _prepare_input_for_model(loaded_model_xgb, best_features_xgb)

    # run predictions per-model and catch errors (e.g., feature mismatch)
    prediction1 = prediction2 = prediction3 = None

    if loaded_model_lr is not None:
        try:
            prediction1 = loaded_model_lr.predict(df_best_features_lr)
        except Exception as e:
            st.warning(f"Linear Regression prediction error: {e}")

    if loaded_model_rfr is not None:
        try:
            prediction2 = loaded_model_rfr.predict(df_best_features_rfr)
        except Exception as e:
            st.warning(f"Random Forest prediction error: {e}")

    if loaded_model_xgb is not None:
        try:
            prediction3 = loaded_model_xgb.predict(df_best_features_xgb)
        except Exception as e:
            st.warning(f"XGBoost prediction error: {e}")

    # If no model produced a prediction, try fallback to root model if present
    if prediction1 is None and prediction2 is None and prediction3 is None:
        try:
            fallback = _safe_load_pickle("house_price_prediction_model.sav")
            if fallback is not None:
                # fallback expects raw input array in original app format
                arr = np.asarray(input_data).reshape(1, -1)
                pred = fallback.predict(arr)
                st.info(f"Fallback model prediction: {pred[0]:.2f} $")
        except Exception as e:
            st.error(f"No available model could produce a prediction: {e}")

    return prediction1, prediction2, prediction3

    
  
def main():
    
    #giving a title
    st.title('House Price Prediction Web App')
    
    col1 , col2 , col3 = st.columns(3)
    #getting input data from user
    with col1:
        SquareMeters = st.number_input("Size of house in square meters")
    with col2:
        NumberOfRooms = st.number_input("Number Of Rooms")
    with col3:
        option1 = st.selectbox('Has Yard',('No', 'Yes')) 
        HasYard = 1 if option1 == 'Yes' else 0
    with col1:
        option2 = st.selectbox('Has Pool',('No', 'Yes')) 
        HasPool = 1 if option2 == 'Yes' else 0
    with col2:
        Floors = st.number_input("Number of floors")
    with col3:
        CityCode = st.number_input("City Code")
    with col1:
        CityPartRange = st.selectbox('City Part Range(cheapest to expensive)',('1','2','3','4','5','6','7','8','9','10')) 
    with col2:
        NumPrevOwners = st.selectbox('Number Prev Owners',('1','2','3','4','5','6','7','8','9','10'))
    with col3:
        Made = st.number_input("Made in Year")
    with col1:
        option3 = st.selectbox('Is New Built',('No', 'Yes'))
        IsNewBuilt = 1 if option3 == 'Yes' else 0
    with col2:
        option4 = st.selectbox('Has Storm Protector',('No', 'Yes'))
        HasStormProtector = 1 if option4 == 'Yes' else 0
    with col3:
        Basement = st.number_input('Basement in square meters')
    with col1:
        Attic = st.number_input('Attic in square meteres')
    with col2:
        Garage = st.number_input('Garage Size in square meteres')
    with col3:
        option5 = st.selectbox('Has Storage Room',('No', 'Yes'))
        HasStorageRoom = 1 if option5 == 'Yes' else 0
    with col1:
        HasGuestRoom = st.number_input('Number of guest rooms')	
    
    
    # code for prediction
    house_price_prediction_lr = ''
    house_price_prediction_rfr = ''
    house_price_prediction_xgb = ''
    

    house_price_prediction_lr,house_price_prediction_rfr,house_price_prediction_xgb = house_price_prediction([SquareMeters,NumberOfRooms,HasYard,HasPool,Floors,CityCode,CityPartRange,NumPrevOwners,Made,IsNewBuilt,HasStormProtector,Basement,Attic,Garage,HasStorageRoom,HasGuestRoom])
        
    #creating a button for Prediction
    if st.button("Predict House Price"):
        prediction = house_price_prediction_lr[0]
        prediction = "{:.2f}".format(prediction)
        st.write(f"The Predicted Price: {prediction} $")
    
    if st.checkbox("Show Advanced Options"):
        if st.button("Predict House Price with Linear Regression Model"):
            prediction = house_price_prediction_lr[0]
            prediction = "{:.2f}".format(prediction)
            st.write(f"The Predicted Price: {prediction} $")
        if st.button("Predict House Price with Random Forest Regressor Model"):
            prediction = house_price_prediction_rfr[0]
            prediction = "{:.2f}".format(prediction)
            st.write(f"The Predicted Price: {prediction} $")
        if st.button("Predict House Price with XG Boost Regressor"):
            prediction = house_price_prediction_xgb[0]
            prediction = "{:.2f}".format(prediction)
            st.write(f"The Predicted Price: {prediction} $")   
    
    
    
if __name__ == '__main__':
    main()
    # Footer with copyright
    try:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown('<div style="color:#9CA3AF; font-size:13px">&copy; 2026 Akash Mahawar &nbsp;•&nbsp; Built with ❤</div>', unsafe_allow_html=True)
    except Exception:
        # avoid breaking if Streamlit UI not available
        pass
    
    
