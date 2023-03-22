#!/usr/bin/env python
# coding: utf-8

# In[2]:





# In[6]:


from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained machine learning model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the Bitcoin live stream data into a pandas DataFrame
df = pd.read_csv('accurate.csv')

# Preprocess the data
df['time'] = pd.to_datetime(df['time'])
scaler = StandardScaler()
df['hash_rate'] = scaler.fit_transform(df['hash_rate'].values.reshape(-1,1))
# new addition of route 
@app.route('/')
def home():
    return "Welcome to the Bitcoin Live Stream Dashboard"
# Define the dashboard route
@app.route('/dashboard')
def dashboard():
    # Get the live data and predictions from the machine learning model
    live_data = df.tail(1).values[0]
    prediction = model.predict(live_data.reshape(1,-1))[0]
    # Render the HTML template containing the live data and predictions
    return render_template('dashboard.html', live_data=live_data, prediction=prediction)

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the live data from the POST request
    live_data = request.form['live_data']
    live_data = scaler.transform([float(live_data)])
    # Get the accuracy of the machine learning model for the live data
    accuracy = model.score(live_data.reshape(1,-1))
    # Return the accuracy as a string
    return str(accuracy)

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




