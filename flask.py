from flask import Flask, jsonify, request, render_template
import requests
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Fetch the Bitcoin mining data from Blockchain.com
def fetch_data():
    url = 'https://api.blockchain.info/charts/hash-rate?timespan=1week&format=json'
    response = requests.get(url)
    data = json.loads(response.text)
    df = pd.DataFrame(data['values'], columns=['time', 'hash_rate'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# Preprocess the data
def preprocess_data(df):
    df['hash_rate_diff'] = df['hash_rate'].diff()
    df.dropna(inplace=True)
    return df

# Train a linear regression model on the data
def train_model(df):
    X = df['hash_rate_diff'].values.reshape(-1, 1)
    y = df['hash_rate'].values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    return model

# Create the Flask app
app = Flask(__name__)

# Define the routes
@app.route('/')
def live_dashboard():
    # Fetch the data and preprocess it
    df = fetch_data()
    df = preprocess_data(df)
    
    # Train the model
    model = train_model(df)
    
    # Predict the next hash rate
    next_hash_rate = model.predict([[df['hash_rate_diff'].iloc[-1]]])[0][0]
    
    # Create the plot
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['hash_rate'], name='Hash Rate'))
    fig.add_trace(go.Scatter(x=[df.index[-1], df.index[-1]+pd.Timedelta(hours=1)], y=[df['hash_rate'].iloc[-1], next_hash_rate], name='Next Hash Rate', mode='lines'))
    fig.update_layout(title='Bitcoin Hash Rate', xaxis_title='Time', yaxis_title='Hash Rate (TH/s)', showlegend=True)
    
    # Render the dashboard
    return render_template('dashboard.html', plot=fig.to_html())

@app.route('/predict', methods=['POST'])
def predict():
    # Fetch the data and preprocess it
    df = fetch_data()
    df = preprocess_data(df)
    
    # Train the model
    model = train_model(df)
    
    # Get the input data from the user
    input_data = request.form['input_data']
    
    # Make the prediction using the trained model
    prediction = model.predict([[float(input_data)]])[0][0]
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
