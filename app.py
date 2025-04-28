from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler (outside route for efficiency)
try:
    lgbm_model = joblib.load('lgbm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    # Verify the expected feature order
    expected_features = ['gender', 'age', 'hypertension', 'heart_disease', 
                        'smoking_history', 'bmi', 'hba1c', 'blood_glucose']
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    raise

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate form data
        form_data = {
            'gender': int(request.form['gender']),
            'age': int(request.form['age']),
            'hypertension': int(request.form['hypertension']),
            'heart_disease': int(request.form['heart_disease']),
            'smoking_history': int(request.form['smoking_history']),
            'bmi': float(request.form['bmi']),
            'hba1c': float(request.form['hba1c']),
            'blood_glucose': float(request.form['blood_glucose'])
        }

        # Create input array in consistent feature order
        input_values = np.array([
            form_data['gender'],
            form_data['age'],
            form_data['hypertension'],
            form_data['heart_disease'],
            form_data['smoking_history'],
            form_data['bmi'],
            form_data['hba1c'],
            form_data['blood_glucose']
        ]).reshape(1, -1)

        # Scale the input
        scaled_input = scaler.transform(input_values)

        # Make prediction
        predicted_class = lgbm_model.predict(scaled_input)
        predicted_prob = lgbm_model.predict_proba(scaled_input)

        # Prepare results
        result = "Positive" if predicted_class[0] == 1 else "Negative"
        probability = round(predicted_prob[0][1] * 100, 2)

        return render_template('index.html', 
                             result=result, 
                             probability=probability,
                             form_data=form_data)  # Pass form data back for display

    except ValueError as ve:
        return render_template('index.html', 
                             error=f"Invalid input: {str(ve)}")
    except Exception as e:
        return render_template('index.html', 
                             error=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)