import React from 'react';
import PredictionForm from './components/PredictionForm';
import './App.css'; // Import App.css

function App() {
    return (
        <div className="app-container">
            <h1>Diabetes Prediction</h1>
            <PredictionForm />
        </div>
    );
}

export default App;