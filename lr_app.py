import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler,LabelEncoder
import numpy as np
import pickle
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://kaleem:kaleem1234@cluster0.fpf7f.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))

db = client['student']
collection = db['student_prediction']



def load_model():
    with open('linear_regression_model.pkl','rb') as file:
        model , scaler ,le =  pickle.load(file)
    return model , scaler ,le

def preprocessing_input_data(data,scaler,le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    preprocessed_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(preprocessed_data)
    return round(prediction[0],2)


def main():
    st.title("Student preformance prediction")
    st.write("enter your detail for get prediction on your input")

    hour_studied = st.number_input("Hours studied",min_value=1,max_value= 10 ,value = 5)
    Previous_Scores= st.number_input("Previous Scores",min_value = 40 , max_value= 100 , value = 70)
    Extracurricular_Activities= st.selectbox("Extracurricular Activities",["Yes" ,"No"])
    Sleep_Hours = st.number_input("Sleep Hours" , min_value= 4 , max_value= 10 , value = 7)
    Question_Papers  = st.number_input("Question Papers" , min_value = 0 , max_value= 10 , value = 5 )


    if st.button("predict_your_score"):
        user_data = {
            "Hours Studied":hour_studied,
            "Previous Scores":Previous_Scores,
            "Extracurricular Activities":Extracurricular_Activities,
            "Sleep Hours":Sleep_Hours,
            "Sample Question Papers Practiced":Question_Papers

        }
        prediction = predict_data(user_data)
        st.success(f"your prediction result is:{prediction} ")
        user_data['prediction'] = prediction
        user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value for key, value in user_data.items()}
        collection.insert_one(user_data)

if __name__ == "__main__":
    main()