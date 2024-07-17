from flask import Flask, request, jsonify
import pandas as pd
import joblib
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

try:
    # Load the saved pipeline (including preprocessing and model)
    pipeline = joblib.load('gradient_boosting_model.joblib')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

@app.route('/', methods=['GET'])
def home():
    return '''
        <form action="/predict" method="post">
            Year: <input type="text" name="Year"><br>
            Month: <input type="text" name="Month"><br>
            Day: <input type="text" name="Day"><br>
            Day of the Week: <input type="text" name="Day of the Week"><br>
            Is Holiday: <input type="text" name="Is Holiday"><br>
            PV Penetration (%): <input type="text" name="PV Penetration (%)"><br>
            GDP Growth Rate (%): <input type="text" name="GDP Growth Rate (%)"><br>
            Population: <input type="text" name="Population"><br>
            Loadshedding Stage: <input type="text" name="Loadshedding Stage"><br>
            EV Penetration (Number): <input type="text" name="EV Penetration (Number)"><br>
            Temperature (°C): <input type="text" name="Temperature (°C)"><br>
            Humidity (%): <input type="text" name="Humidity (%)"><br>
            <input type="submit" value="Submit">
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features from the form
        input_features = {key: request.form[key] for key in request.form.keys()}
        
        # Process input data to match the model's expected format
        input_features_processed = {}
        for key, value in input_features.items():
            try:
                input_features_processed[key] = [float(value)]
            except ValueError:
                input_features_processed[key] = [value]

        # Convert processed input data to DataFrame
        input_df = pd.DataFrame.from_dict(input_features_processed)
        
        # Predict using the loaded pipeline
        predicted_energy = pipeline.predict(input_df)[0]
        
        # Return the prediction result
        return jsonify(prediction=predicted_energy)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
