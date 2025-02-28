import React, { useState } from 'react';
import axios from 'axios';
import './PredictionForm.css';

function PredictionForm() {
    const [height, setHeight] = useState('');
    const [weight, setWeight] = useState('');
    const [bmi, setBMI] = useState(''); // Corrected declaration
    const [formData, setFormData] = useState({ // Corrected declaration
        Pregnancies: '',
        Glucose: '',
        BloodPressure: '',
        SkinThickness: '',
        Insulin: '',
        BMI: '',
        DiabetesPedigreeFunction: '',
        Age: '',
    });
    //const [prediction, setPrediction] = useState(null);

    const calculateBMI = () => {
        if (height && weight) {
            const heightInMeters = height / 100;
            const calculatedBMI = weight / (heightInMeters * heightInMeters);
            setBMI(calculatedBMI.toFixed(2));
            setFormData({ ...formData, BMI: calculatedBMI.toFixed(2) });
        }
    };

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const [predictionMessage, setPredictionMessage] = useState(''); // New state variable

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const response = await axios.post('http://127.0.0.1:5000/predict', formData);
            const prediction = response.data.prediction;
            if (prediction === 1) {
                setPredictionMessage("Based on your inputs, there's a higher likelihood of diabetes. It's recommended to consult a healthcare professional for further evaluation.");
            } else {
                setPredictionMessage("Based on your inputs, the likelihood of diabetes appears to be low. However, maintain a healthy lifestyle and consult your doctor for regular check-ups.");
            }
        } catch (error) {
            console.error('Error:', error);
            setPredictionMessage("An error occurred. Please try again.");
        }
    };

    return (
        <div className="prediction-form-container"> {/* Add a container div */}
            <h2 className="section-title">BMI Calculator</h2>
            <div className="bmi-calculator">
                <label htmlFor="height">Height (cm):</label>
                <input type="number" id="height" value={height} onChange={(e) => setHeight(e.target.value)} />
                <label htmlFor="weight">Weight (kg):</label>
                <input type="number" id="weight" value={weight} onChange={(e) => setWeight(e.target.value)} />
                <button type="button" onClick={calculateBMI}>Calculate BMI</button>
                {bmi && <p className="bmi-result">Your BMI: {bmi}</p>}
            </div>

            <h2 className="section-title">Diabetes Prediction</h2>
            <form onSubmit={handleSubmit}>
                {Object.keys(formData).map((key) => (
                    <div key={key} className="form-group">
                        <label htmlFor={key}>{key}:</label>
                        <input
                            type="number"
                            id={key}
                            name={key}
                            value={formData[key]}
                            onChange={handleChange}
                        />
                    </div>
                ))}
                <button type="submit" className="predict-button">Predict</button>
            </form>
            {predictionMessage && <p className="prediction-message">{predictionMessage}</p>}
        </div>
    );
}

export default PredictionForm;