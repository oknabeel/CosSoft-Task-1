import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Load the trained model from the .sav file
model = joblib.load('model.pkl')

st.title('Credit Card Fraud Detection')

# Create input fields for user to enter transaction details
cc_num = st.text_input('Credit Card Number')
amt = st.number_input('Transaction Amount')
zip_code = st.number_input('ZIP Code')
merchant = st.text_input('Merchant')
category = st.text_input('Category')
first = st.text_input('First Name')
last = st.text_input('Last Name')
gender = st.text_input('Gender')
street = st.text_input('Street')
city = st.text_input('City')
state = st.text_input('State')

# Preprocess the input data and make a prediction
input_data = pd.DataFrame({
    'cc_num': [cc_num],
    'amt': [amt],
    'zip': [zip_code],
    'merchant': [merchant],
    'category': [category],
    'first': [first],
    'last': [last],
    'gender': [gender],
    'street': [street],
    'city': [city],
    'state': [state]
})

#======================================================================================
def set_input(label):
    
    columns_to_encode = ['merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state']

    for column in columns_to_encode:
        input_data[column + '_encoded'] = le.fit_transform(input_data[column])
    df_encoded = input_data.drop(columns=columns_to_encode)

    return df_encoded

new_df = set_input(input_data)
#======================================================================================

if st.button('Predict'):
    # Make predictions
    prediction = model.predict(new_df)[0]
    
    # Display the prediction result
    st.subheader('Transaction Prediction:')
    if prediction == 1:
        st.write('The transaction is predicted as fraudulent.')
    else:
        st.write('The transaction is predicted as legitimate.')
