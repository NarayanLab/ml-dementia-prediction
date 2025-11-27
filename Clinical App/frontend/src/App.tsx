import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

// API URL - defaults to localhost for local development
// For deployment, set REACT_APP_API_URL environment variable
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

interface PatientData {
  af_age: number;
  marital_status: string;
  weight: number;
  bmi: number;
  diabetes: boolean;
  hypertension: boolean;
  stroke_tia: boolean;
  depression: boolean;
  cognitive_impairment: boolean;
  rr_interval: number;
  qrs_duration: number;
  sodium_value: number;
  calcium_mg_dl: number;
  osteoarthritis: boolean;
  race: number;
  insurance: number;
  osteoporosis: boolean;
  parkinson: boolean;
}

interface FormInputs {
  af_age: number;
  marital_status: string;
  bmi: number;
  weight: number;
  height: number;
  diabetes: boolean;
  hypertension: boolean;
  stroke_tia: boolean;
  depression: boolean;
  cognitive_impairment: boolean;
  hr_method: 'heart_rate' | 'rr_interval';
  heart_rate: number;
  rr_interval: number;
  qrs_duration: number;
  sodium_value: number;
  calcium_unit: 'mg_dl' | 'mmol_l';
  calcium_mg_dl: number;
  calcium_mmol_l: number;
  osteoarthritis: boolean;
  race: string;
  insurance: string;
  osteoporosis: boolean;
  parkinson: boolean;
}

interface RiskResponse {
  risk_percentage: number;
  risk_category: string;
  risk_color: string;
  low_threshold: number;
  high_threshold: number;
}

// Component definitions (moved outside App to prevent re-creation on each render)
const ToggleSwitch: React.FC<{
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
}> = ({ label, value, onChange }) => (
  <div className="toggle-switch-group">
    <label className="toggle-switch-label">{label}</label>
    <div className="toggle-switch-container">
      <span className={`toggle-text ${!value ? 'active' : ''}`}>No</span>
      <div
        className={`toggle-switch ${value ? 'on' : 'off'}`}
        onClick={() => onChange(!value)}
      >
        <div className="toggle-slider" />
      </div>
      <span className={`toggle-text ${value ? 'active' : ''}`}>Yes</span>
    </div>
  </div>
);

const NumberInput: React.FC<{
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  unit?: string;
  min?: number;
  max?: number;
}> = ({ label, value, onChange, step = 1, unit, min, max }) => (
  <div className="input-group">
    <label className="input-label">{label} {unit && <span className="unit">({unit})</span>}</label>
    <input
      type="number"
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
      step={step}
      min={min}
      max={max}
      className="number-input"
    />
  </div>
);

const SelectInput: React.FC<{
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}> = ({ label, value, options, onChange }) => (
  <div className="input-group">
    <label className="input-label">{label}</label>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="select-input"
    >
      {options.map(option => (
        <option key={option} value={option}>{option}</option>
      ))}
    </select>
  </div>
);

const CompactToggle: React.FC<{
  label: string;
  options: { value: string; label: string }[];
  value: string;
  onChange: (value: string) => void;
}> = ({ label, options, value, onChange }) => (
  <div className="compact-toggle-group">
    <label className="input-label">{label}</label>
    <div className="compact-toggle-buttons">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className={`compact-toggle-btn ${value === option.value ? 'active' : ''}`}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  </div>
);

const RadioButtons: React.FC<{
  label: string;
  options: string[];
  value: string;
  onChange: (value: string) => void;
}> = ({ label, options, value, onChange }) => (
  <div className="radio-buttons-group">
    <label className="input-label">{label}</label>
    <div className="radio-buttons-container">
      {options.map((option) => (
        <label key={option} className="radio-button-option">
          <input
            type="radio"
            value={option}
            checked={value === option}
            onChange={() => onChange(option)}
          />
          <span className="radio-button-label">{option}</span>
        </label>
      ))}
    </div>
  </div>
);

const App: React.FC = () => {
  const [formData, setFormData] = useState<FormInputs>({
    af_age: 70,
    marital_status: 'Married',
    bmi: 27.2,
    weight: 79.3,
    height: 170.0,
    diabetes: false,
    hypertension: false,
    stroke_tia: false,
    depression: false,
    cognitive_impairment: false,
    hr_method: 'heart_rate',
    heart_rate: 77,
    rr_interval: 778,
    qrs_duration: 100,
    sodium_value: 138.5,
    calcium_unit: 'mg_dl',
    calcium_mg_dl: 9.6,
    calcium_mmol_l: 2.4,
    osteoarthritis: false,
    race: 'White',
    insurance: 'Public',
    osteoporosis: false,
    parkinson: false
  });

  const [result, setResult] = useState<RiskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  const handleInputChange = (field: keyof FormInputs, value: any) => {
    setFormData(prev => {
      const newData = { ...prev, [field]: value };

      // Auto-calculate BMI when height/weight changes
      if (field === 'weight' || field === 'height') {
        newData.bmi = Math.round(calculateBMI(newData.weight, newData.height) * 10) / 10;
      }

      // Auto-calculate RR interval when heart rate changes
      if (field === 'heart_rate' && newData.hr_method === 'heart_rate') {
        newData.rr_interval = Math.round(heartRateToRRInterval(newData.heart_rate));
      }

      // Auto-calculate calcium conversions
      if (field === 'calcium_mg_dl' && newData.calcium_unit === 'mg_dl') {
        newData.calcium_mmol_l = newData.calcium_mg_dl / 4.008;
      }
      if (field === 'calcium_mmol_l' && newData.calcium_unit === 'mmol_l') {
        newData.calcium_mg_dl = newData.calcium_mmol_l * 4.008;
      }

      return newData;
    });
  };

  const calculateBMI = (weight: number, height: number): number => {
    if (height <= 0) return 0;
    const heightInMeters = height / 100;
    return weight / (heightInMeters * heightInMeters);
  };

  const heartRateToRRInterval = (heartRate: number): number => {
    if (heartRate <= 0) return 0;
    return 60000 / heartRate;
  };

  const calculateRisk = async () => {
    setLoading(true);
    setError('');

    try {
      // Map race and insurance to numeric codes
      // Race: 0=White, 1=Black, 2=Other/Unknown
      const raceMap: { [key: string]: number } = {
        'White': 0,
        'Black': 1,
        'Other/Unknown': 2
      };

      // Insurance: 0=Public (Medicare/Medicaid), 1=Private, 2=Unknown
      const insuranceMap: { [key: string]: number } = {
        'Public': 0,
        'Private': 1,
        'Unknown': 2
      };

      // Prepare data for API
      const apiData: PatientData = {
        af_age: formData.af_age,
        marital_status: formData.marital_status,
        weight: formData.weight,
        bmi: calculateBMI(formData.weight, formData.height),
        diabetes: formData.diabetes,
        hypertension: formData.hypertension,
        stroke_tia: formData.stroke_tia,
        depression: formData.depression,
        cognitive_impairment: formData.cognitive_impairment,
        rr_interval: formData.hr_method === 'heart_rate' ? Math.round(heartRateToRRInterval(formData.heart_rate)) : formData.rr_interval,
        qrs_duration: formData.qrs_duration,
        sodium_value: formData.sodium_value,
        calcium_mg_dl: formData.calcium_unit === 'mg_dl' ? formData.calcium_mg_dl : formData.calcium_mmol_l * 4.008,
        osteoarthritis: formData.osteoarthritis,
        race: raceMap[formData.race] || 0,
        insurance: insuranceMap[formData.insurance] || 0,
        osteoporosis: formData.osteoporosis,
        parkinson: formData.parkinson
      };

      const response = await axios.post<RiskResponse>(`${API_URL}/predict`, apiData);
      setResult(response.data);
    } catch (err) {
      setError('Failed to calculate risk. Please check your inputs and try again.');
      console.error('API Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getRiskBarWidth = () => {
    if (!result) return 0;
    return Math.min(result.risk_percentage, 100);
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>Dementia Risk Assessment</h1>
        </header>

        <div className="form-section">
          <div className="single-column-form">

            {/* Age at AF Diagnosis */}
            <div className="form-row">
              <NumberInput
                label="Age at AF Diagnosis"
                value={formData.af_age}
                onChange={(value) => handleInputChange('af_age', value)}
                unit="years"
                min={18}
                max={110}
              />
            </div>

            {/* Marital Status */}
            <div className="form-row">
              <RadioButtons
                label="Marital Status"
                options={['Single', 'Married', 'Divorced/Widowed', 'Unknown']}
                value={formData.marital_status}
                onChange={(value) => handleInputChange('marital_status', value)}
              />
            </div>

            {/* Weight and Height */}
            <div className="form-row dual-input">
              <NumberInput
                label="Weight"
                value={formData.weight}
                onChange={(value) => handleInputChange('weight', value)}
                unit="kg"
                step={0.1}
                min={30}
                max={250}
              />
              <NumberInput
                label="Height"
                value={formData.height}
                onChange={(value) => handleInputChange('height', value)}
                unit="cm"
                step={0.5}
                min={100}
                max={250}
              />
              <div className="calculated-value">
                <span>Calculated BMI: {calculateBMI(formData.weight, formData.height).toFixed(1)} kg/m²</span>
              </div>
            </div>

            {/* Medical History */}
            <div className="form-row">
              <h3 className="section-divider">Medical History</h3>
            </div>

            <div className="form-row">
              <ToggleSwitch
                label="Diabetes Mellitus"
                value={formData.diabetes}
                onChange={(value) => handleInputChange('diabetes', value)}
              />
            </div>

            <div className="form-row">
              <ToggleSwitch
                label="Hypertension"
                value={formData.hypertension}
                onChange={(value) => handleInputChange('hypertension', value)}
              />
            </div>

            <div className="form-row">
              <ToggleSwitch
                label="History of Stroke/TIA"
                value={formData.stroke_tia}
                onChange={(value) => handleInputChange('stroke_tia', value)}
              />
            </div>

            <div className="form-row">
              <ToggleSwitch
                label="Depression"
                value={formData.depression}
                onChange={(value) => handleInputChange('depression', value)}
              />
            </div>

            <div className="form-row">
              <ToggleSwitch
                label="Cognitive Impairment"
                value={formData.cognitive_impairment}
                onChange={(value) => handleInputChange('cognitive_impairment', value)}
              />
            </div>

            <div className="form-row">
              <ToggleSwitch
                label="Osteoarthritis"
                value={formData.osteoarthritis}
                onChange={(value) => handleInputChange('osteoarthritis', value)}
              />
            </div>

            <div className="form-row">
              <ToggleSwitch
                label="Osteoporosis"
                value={formData.osteoporosis}
                onChange={(value) => handleInputChange('osteoporosis', value)}
              />
            </div>

            <div className="form-row">
              <ToggleSwitch
                label="Parkinson's Disease"
                value={formData.parkinson}
                onChange={(value) => handleInputChange('parkinson', value)}
              />
            </div>

            {/* Demographics */}
            <div className="form-row">
              <h3 className="section-divider">Demographics</h3>
            </div>

            <div className="form-row">
              <RadioButtons
                label="Race"
                options={['White', 'Black', 'Other/Unknown']}
                value={formData.race}
                onChange={(value) => handleInputChange('race', value)}
              />
            </div>

            <div className="form-row">
              <RadioButtons
                label="Insurance Type"
                options={['Public', 'Private', 'Unknown']}
                value={formData.insurance}
                onChange={(value) => handleInputChange('insurance', value)}
              />
            </div>

            {/* Clinical Values */}
            <div className="form-row">
              <h3 className="section-divider">Clinical Values</h3>
            </div>

            {/* Heart Rate / RR Interval */}
            <div className="form-row">
              <CompactToggle
                label="Heart Rate / RR Interval"
                options={[
                  { value: 'heart_rate', label: 'Heart Rate (bpm)' },
                  { value: 'rr_interval', label: 'RR Interval (ms)' }
                ]}
                value={formData.hr_method}
                onChange={(value) => handleInputChange('hr_method', value as 'heart_rate' | 'rr_interval')}
              />
            </div>

            {formData.hr_method === 'heart_rate' ? (
              <div className="form-row">
                <NumberInput
                  label="Heart Rate"
                  value={formData.heart_rate}
                  onChange={(value) => handleInputChange('heart_rate', value)}
                  unit="bpm"
                  min={30}
                  max={200}
                />
                <div className="calculated-value">
                  <span>Calculated RR Interval: {heartRateToRRInterval(formData.heart_rate).toFixed(0)} ms</span>
                </div>
              </div>
            ) : (
              <div className="form-row">
                <NumberInput
                  label="RR Interval"
                  value={formData.rr_interval}
                  onChange={(value) => handleInputChange('rr_interval', Math.round(value))}
                  unit="ms"
                  min={300}
                  max={2000}
                />
              </div>
            )}

            <div className="form-row">
              <NumberInput
                label="QRS Axis"
                value={formData.qrs_duration}
                onChange={(value) => handleInputChange('qrs_duration', value)}
                unit="degrees"
                min={60}
                max={200}
              />
            </div>

            <div className="form-row">
              <NumberInput
                label="Sodium"
                value={formData.sodium_value}
                onChange={(value) => handleInputChange('sodium_value', value)}
                unit="mmol/L"
                step={0.1}
                min={120}
                max={160}
              />
            </div>

            {/* Calcium */}
            <div className="form-row">
              <CompactToggle
                label="Calcium"
                options={[
                  { value: 'mg_dl', label: 'mg/dL' },
                  { value: 'mmol_l', label: 'mmol/L' }
                ]}
                value={formData.calcium_unit}
                onChange={(value) => handleInputChange('calcium_unit', value as 'mg_dl' | 'mmol_l')}
              />
            </div>

            {formData.calcium_unit === 'mg_dl' ? (
              <div className="form-row">
                <NumberInput
                  label="Calcium"
                  value={formData.calcium_mg_dl}
                  onChange={(value) => handleInputChange('calcium_mg_dl', value)}
                  unit="mg/dL"
                  step={0.1}
                  min={6}
                  max={16}
                />
              </div>
            ) : (
              <div className="form-row">
                <NumberInput
                  label="Calcium"
                  value={formData.calcium_mmol_l}
                  onChange={(value) => handleInputChange('calcium_mmol_l', value)}
                  unit="mmol/L"
                  step={0.01}
                  min={1.5}
                  max={4.0}
                />
              </div>
            )}

          </div>

          <button
            onClick={calculateRisk}
            disabled={loading}
            className="calculate-button"
          >
            {loading ? 'Calculating...' : 'Calculate Risk'}
          </button>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          {result && (
            <div className="results-section">
              <div className="risk-display">
                <div className="risk-header">
                  <span className="risk-label">5-Year Dementia Risk</span>
                  <span
                    className="risk-value"
                    style={{ color: result.risk_color }}
                  >
                    {result.risk_percentage.toFixed(1)}%
                  </span>
                </div>

                <div className="risk-bar-container">
                  <div className="risk-bar-background">
                    <div className="risk-zones">
                      <div
                        className="risk-zone low"
                        style={{ width: `${result.low_threshold}%` }}
                      />
                      <div
                        className="risk-zone medium"
                        style={{
                          width: `${result.high_threshold - result.low_threshold}%`,
                          left: `${result.low_threshold}%`
                        }}
                      />
                      <div
                        className="risk-zone high"
                        style={{
                          width: `${100 - result.high_threshold}%`,
                          left: `${result.high_threshold}%`
                        }}
                      />
                    </div>
                    <div
                      className="risk-indicator"
                      style={{
                        left: `${getRiskBarWidth()}%`,
                        backgroundColor: result.risk_color
                      }}
                    />
                  </div>
                  <div className="risk-labels">
                    <span>0%</span>
                    <span>{result.low_threshold.toFixed(0)}%</span>
                    <span>{result.high_threshold.toFixed(0)}%</span>
                    <span>100%</span>
                  </div>
                </div>

                <div
                  className="risk-category"
                  style={{ color: result.risk_color }}
                >
                  {result.risk_category}
                </div>
              </div>
            </div>
          )}
        </div>

        <footer className="footer">
          <p>XGBoost-Cox model for 5-year dementia risk prediction in AF patients</p>
          <p>For clinical decision support only. Not intended as sole basis for medical decisions.</p>
        </footer>
      </div>
    </div>
  );
};

export default App;
