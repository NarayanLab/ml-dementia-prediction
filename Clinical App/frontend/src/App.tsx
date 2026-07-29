import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Population means (for imputing unavailable lab values)
const MEANS = {
  sodium: 138.5,
  potassium: 4.2,
  creatinine: 1.13,
  calcium_mg_dl: 8.6,
  rr_interval: 778,
  heart_rate: 77,
  qrs: 17,
};

interface PatientData {
  af_age: number;
  marital_status: string;
  weight: number;
  height: number;
  bmi: number;
  diabetes: boolean;
  hypertension: boolean;
  stroke_tia: boolean;
  depression: boolean;
  cognitive_deficit: boolean;
  osteoarthritis: boolean;
  parkinson: boolean;
  ppi: boolean;
  insurance: number;
  rr_interval: number;
  qrs_duration: number;
  sodium_value: number;
  potassium_value: number;
  creatinine_value: number;
  calcium_mg_dl: number;
  calcium_available: boolean;
  hct_available: boolean;
}

interface Availability {
  calcium: boolean;
  hct: boolean;
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
  cognitive_deficit: boolean;
  osteoarthritis: boolean;
  parkinson: boolean;
  ppi: boolean;
  hr_method: 'heart_rate' | 'rr_interval';
  heart_rate: number;
  rr_interval: number;
  qrs_duration: number;
  sodium_value: number;
  potassium_value: number;
  creatinine_value: number;
  calcium_unit: 'mg_dl' | 'mmol_l';
  calcium_mg_dl: number;
  calcium_mmol_l: number;
  hct_value: number;
  insurance: string;
  availability: Availability;
}

interface RiskResponse {
  risk_percentage: number;
  risk_category: string;
  risk_color: string;
  low_threshold: number;
  high_threshold: number;
}

const ToggleSwitch: React.FC<{
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
  onLabel?: string;
  offLabel?: string;
}> = ({ label, value, onChange, onLabel = 'Yes', offLabel = 'No' }) => (
  <div className="toggle-switch-group">
    <label className="toggle-switch-label">{label}</label>
    <div className="toggle-switch-container">
      <span className={`toggle-text ${!value ? 'active' : ''}`}>{offLabel}</span>
      <div
        className={`toggle-switch ${value ? 'on' : 'off'}`}
        onClick={() => onChange(!value)}
      >
        <div className="toggle-slider" />
      </div>
      <span className={`toggle-text ${value ? 'active' : ''}`}>{onLabel}</span>
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
  disabled?: boolean;
}> = ({ label, value, onChange, step = 1, unit, min, max, disabled }) => {
  const [displayValue, setDisplayValue] = React.useState<string>(String(value));

  React.useEffect(() => {
    setDisplayValue(String(value));
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const rawValue = e.target.value;
    setDisplayValue(rawValue);
    const parsed = parseFloat(rawValue);
    if (!isNaN(parsed)) {
      onChange(parsed);
    }
  };

  const handleBlur = () => {
    const parsed = parseFloat(displayValue);
    if (isNaN(parsed) || displayValue === '') {
      setDisplayValue('0');
      onChange(0);
    }
  };

  return (
    <div className="input-group">
      <label className="input-label">{label} {unit && <span className="unit">({unit})</span>}</label>
      <input
        type="number"
        value={displayValue}
        onChange={handleChange}
        onBlur={handleBlur}
        step={step}
        min={min}
        max={max}
        disabled={disabled}
        className="number-input"
      />
    </div>
  );
};

const CompactToggle: React.FC<{
  label?: string;
  options: { value: string; label: string }[];
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}> = ({ label, options, value, onChange, disabled }) => (
  <div className={`compact-toggle-group ${disabled ? 'disabled' : ''}`}>
    {label && <label className="input-label">{label}</label>}
    <div className="compact-toggle-buttons">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className={`compact-toggle-btn ${value === option.value ? 'active' : ''}`}
          onClick={() => !disabled && onChange(option.value)}
          disabled={disabled}
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
    cognitive_deficit: false,
    osteoarthritis: false,
    parkinson: false,
    ppi: false,
    hr_method: 'heart_rate',
    heart_rate: 77,
    rr_interval: 778,
    qrs_duration: 17,
    sodium_value: 138.5,
    potassium_value: 4.2,
    creatinine_value: 1.13,
    calcium_unit: 'mg_dl',
    calcium_mg_dl: 8.6,
    calcium_mmol_l: 2.15,
    hct_value: 40.0,
    insurance: 'Public',
    availability: {
      calcium: true,
      hct: true,
    },
  });

  const [result, setResult] = useState<RiskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  const handleInputChange = (field: keyof FormInputs, value: any) => {
    setFormData(prev => {
      const newData = { ...prev, [field]: value } as FormInputs;

      if (field === 'weight' || field === 'height') {
        newData.bmi = Math.round(calculateBMI(newData.weight, newData.height) * 10) / 10;
      }

      if (field === 'heart_rate' && newData.hr_method === 'heart_rate') {
        newData.rr_interval = Math.round(heartRateToRRInterval(newData.heart_rate));
      }

      if (field === 'calcium_mg_dl' && newData.calcium_unit === 'mg_dl') {
        newData.calcium_mmol_l = newData.calcium_mg_dl / 4.008;
      }
      if (field === 'calcium_mmol_l' && newData.calcium_unit === 'mmol_l') {
        newData.calcium_mg_dl = newData.calcium_mmol_l * 4.008;
      }

      return newData;
    });
  };

  const setAvailability = (key: keyof Availability, available: boolean) => {
    setFormData(prev => ({
      ...prev,
      availability: { ...prev.availability, [key]: available },
    }));
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
      const insuranceMap: { [key: string]: number } = {
        'Public': 0,
        'Private': 1,
        'Unknown': 2,
      };

      const av = formData.availability;

      const rr = formData.hr_method === 'heart_rate'
        ? Math.round(heartRateToRRInterval(formData.heart_rate))
        : formData.rr_interval;

      const calciumMgDl = av.calcium
        ? (formData.calcium_unit === 'mg_dl'
            ? formData.calcium_mg_dl
            : formData.calcium_mmol_l * 4.008)
        : MEANS.calcium_mg_dl;

      const apiData: PatientData = {
        af_age: formData.af_age,
        marital_status: formData.marital_status,
        weight: formData.weight,
        height: formData.height,
        bmi: calculateBMI(formData.weight, formData.height),
        diabetes: formData.diabetes,
        hypertension: formData.hypertension,
        stroke_tia: formData.stroke_tia,
        depression: formData.depression,
        cognitive_deficit: formData.cognitive_deficit,
        osteoarthritis: formData.osteoarthritis,
        parkinson: formData.parkinson,
        ppi: formData.ppi,
        insurance: insuranceMap[formData.insurance] || 0,
        rr_interval: rr,
        qrs_duration: formData.qrs_duration,
        sodium_value: formData.sodium_value,
        potassium_value: formData.potassium_value,
        creatinine_value: formData.creatinine_value,
        calcium_mg_dl: calciumMgDl,
        calcium_available: av.calcium,
        hct_available: av.hct,
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

  const av = formData.availability;

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>Stanford Dementia in AF Risk Calculator</h1>
        </header>

        <div className="form-section">
          <div className="single-column-form">

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

            <div className="form-row">
              <RadioButtons
                label="Marital Status"
                options={['Single', 'Married', 'Divorced/Widowed', 'Unknown']}
                value={formData.marital_status}
                onChange={(value) => handleInputChange('marital_status', value)}
              />
            </div>

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
                label="Cognitive Deficit"
                value={formData.cognitive_deficit}
                onChange={(value) => handleInputChange('cognitive_deficit', value)}
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
                label="Parkinson's Disease"
                value={formData.parkinson}
                onChange={(value) => handleInputChange('parkinson', value)}
              />
            </div>

            <div className="form-row">
              <h3 className="section-divider">Medications</h3>
            </div>

            <div className="form-row">
              <ToggleSwitch
                label="PPI (Proton Pump Inhibitor)"
                value={formData.ppi}
                onChange={(value) => handleInputChange('ppi', value)}
              />
            </div>

            <div className="form-row">
              <h3 className="section-divider">Demographics</h3>
            </div>

            <div className="form-row">
              <RadioButtons
                label="Insurance Type"
                options={['Public', 'Private', 'Unknown']}
                value={formData.insurance}
                onChange={(value) => handleInputChange('insurance', value)}
              />
            </div>

            <div className="form-row">
              <h3 className="section-divider">Clinical Values</h3>
            </div>

            {/* Heart Rate / RR Interval */}
            <div className="form-row">
              <CompactToggle
                label="Heart Rate / RR Interval"
                options={[
                  { value: 'heart_rate', label: 'Heart Rate (bpm)' },
                  { value: 'rr_interval', label: 'RR Interval (ms)' },
                ]}
                value={formData.hr_method}
                onChange={(value) => handleInputChange('hr_method', value as 'heart_rate' | 'rr_interval')}
              />
            </div>

            {formData.hr_method === 'heart_rate' ? (
              <div className="form-row">
                <NumberInput
                  label="Heart Rate" unit="bpm" min={30} max={200}
                  value={formData.heart_rate}
                  onChange={(v) => handleInputChange('heart_rate', v)}
                />
                <div className="calculated-value">
                  <span>Calculated RR Interval: {heartRateToRRInterval(formData.heart_rate).toFixed(0)} ms</span>
                </div>
              </div>
            ) : (
              <div className="form-row">
                <NumberInput
                  label="RR Interval" unit="ms" min={300} max={2000}
                  value={formData.rr_interval}
                  onChange={(v) => handleInputChange('rr_interval', Math.round(v))}
                />
              </div>
            )}

            <div className="form-row">
              <NumberInput
                label="QRS Axis" unit="degrees" min={-180} max={180}
                value={formData.qrs_duration}
                onChange={(v) => handleInputChange('qrs_duration', v)}
              />
            </div>

            <div className="form-row">
              <NumberInput
                label="Sodium" unit="mmol/L" step={0.1} min={120} max={160}
                value={formData.sodium_value}
                onChange={(v) => handleInputChange('sodium_value', v)}
              />
            </div>

            <div className="form-row">
              <NumberInput
                label="Potassium" unit="mmol/L" step={0.1} min={2.5} max={7.0}
                value={formData.potassium_value}
                onChange={(v) => handleInputChange('potassium_value', v)}
              />
            </div>

            <div className="form-row">
              <NumberInput
                label="Creatinine" unit="mg/dL" step={0.01} min={0.1} max={15}
                value={formData.creatinine_value}
                onChange={(v) => handleInputChange('creatinine_value', v)}
              />
            </div>

            {/* Calcium — has availability toggle (Calcium_missing is a model feature) */}
            <div className="form-row">
              <ToggleSwitch
                label="Calcium availability"
                value={av.calcium}
                onChange={(v) => setAvailability('calcium', v)}
                onLabel="Available"
                offLabel="Not available"
              />
            </div>
            <div className="form-row">
              <CompactToggle
                label="Calcium"
                options={[
                  { value: 'mg_dl', label: 'mg/dL' },
                  { value: 'mmol_l', label: 'mmol/L' },
                ]}
                value={formData.calcium_unit}
                onChange={(value) => handleInputChange('calcium_unit', value as 'mg_dl' | 'mmol_l')}
                disabled={!av.calcium}
              />
              {formData.calcium_unit === 'mg_dl' ? (
                <NumberInput
                  label="Calcium" unit="mg/dL" step={0.1} min={6} max={16}
                  value={formData.calcium_mg_dl}
                  onChange={(v) => handleInputChange('calcium_mg_dl', v)}
                  disabled={!av.calcium}
                />
              ) : (
                <NumberInput
                  label="Calcium" unit="mmol/L" step={0.01} min={1.5} max={4.0}
                  value={formData.calcium_mmol_l}
                  onChange={(v) => handleInputChange('calcium_mmol_l', v)}
                  disabled={!av.calcium}
                />
              )}
            </div>

            {/* Hematocrit — model uses HCT_missing flag only; value collected for UX completeness */}
            <div className="form-row">
              <ToggleSwitch
                label="Hematocrit availability"
                value={av.hct}
                onChange={(v) => setAvailability('hct', v)}
                onLabel="Available"
                offLabel="Not available"
              />
            </div>
            {av.hct && (
              <div className="form-row">
                <NumberInput
                  label="Hematocrit" unit="%" step={0.1} min={15} max={65}
                  value={formData.hct_value}
                  onChange={(v) => handleInputChange('hct_value', v)}
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
                    <span>25%</span>
                    <span>50%</span>
                    <span>75%</span>
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
          <p className="disclaimer">This platform and all associated software, algorithms, visualizations, and outputs are provided for research and informational purposes only. They are not intended for clinical use, not intended to diagnose, treat, cure, or prevent any disease, and have not been reviewed or approved by the U.S. Food and Drug Administration (FDA), the UK Medicines and Healthcare products Regulatory Agency (MHRA), or any other regulatory authority.</p>
        </footer>
      </div>
    </div>
  );
};

export default App;
