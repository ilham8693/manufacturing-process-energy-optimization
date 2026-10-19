"""
Process Energy Optimization, 3-Phase Power Quality, Thermodynamics & SciPy Setpoint Engine.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import minimize

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ENERGY_SURROGATE_PATH = os.path.join(MODELS_DIR, "energy_surrogate_model.pkl")

# Carbon Emission Factor (e.g. 0.45 kg CO2 per kWh - US/EU industrial grid average)
DEFAULT_EMISSION_FACTOR = 0.45 
DEFAULT_ELECTRICITY_TARIFF = 0.14 # USD / kWh

ENERGY_FEATURES = [
    'Shaft_Speed_RPM', 'Load_Torque', 'Workload_Percentage',
    'Bearing_Temperature', 'Motor_Temperature', 'Oil_Viscosity', 'RMS_Vibration'
]

def calculate_power_quality_metrics(row):
    """
    Computes 3-Phase Voltage Unbalance Factor (VUF), Current Unbalance (IUF),
    Apparent Power (kVA), Reactive Power (kVAR), and Power Factor status.
    
    Standards:
    - IEEE 141 / EN 50160: VUF < 2.0% is optimal/normal; >= 2.0% causes severe motor derating.
    - IUF < 5.0% is balanced.
    """
    va = float(row.get('Voltage_Phase_A', 220.0))
    vb = float(row.get('Voltage_Phase_B', 220.0))
    vc = float(row.get('Voltage_Phase_C', 220.0))
    
    ia = float(row.get('Current_Phase_A', 20.0))
    ib = float(row.get('Current_Phase_B', 20.0))
    ic = float(row.get('Current_Phase_C', 20.0))
    
    pf = float(row.get('Power_Factor', 0.88))
    pf = max(0.01, min(1.0, pf))
    
    v_avg = (va + vb + vc) / 3.0
    max_v_dev = max(abs(va - v_avg), abs(vb - v_avg), abs(vc - v_avg))
    vuf = (max_v_dev / v_avg) * 100.0 if v_avg > 0 else 0.0
    
    i_avg = (ia + ib + ic) / 3.0
    max_i_dev = max(abs(ia - i_avg), abs(ib - i_avg), abs(ic - i_avg))
    iuf = (max_i_dev / i_avg) * 100.0 if i_avg > 0 else 0.0
    
    # Apparent power (kVA) & Reactive Power (kVAR)
    p_kw = float(row.get('Power_Consumption', 500.0))
    s_kva = p_kw / pf
    q_kvar = np.sqrt(max(0.0, s_kva**2 - p_kw**2))
    
    # Motor thermal loss estimation
    eff_index = float(row.get('Energy_Efficiency_Index', 85.0))
    thermal_loss_kw = p_kw * max(0.0, (1.0 - eff_index / 100.0))
    
    return {
        "V_avg": round(v_avg, 1),
        "V_Phase_A": round(va, 1),
        "V_Phase_B": round(vb, 1),
        "V_Phase_C": round(vc, 1),
        "VUF_percent": round(vuf, 2),
        "VUF_status": "Balanced (Optimal <2%)" if vuf < 2.0 else "⚠️ High Unbalance (Derating Required)",
        "I_avg": round(i_avg, 1),
        "I_Phase_A": round(ia, 1),
        "I_Phase_B": round(ib, 1),
        "I_Phase_C": round(ic, 1),
        "IUF_percent": round(iuf, 2),
        "IUF_status": "Balanced (<5%)" if iuf < 5.0 else "⚠️ Unbalanced",
        "Active_Power_kW": round(p_kw, 1),
        "Apparent_Power_kVA": round(s_kva, 1),
        "Reactive_Power_kVAR": round(q_kvar, 1),
        "Power_Factor": round(pf, 3),
        "Thermal_Loss_kW": round(thermal_loss_kw, 2)
    }

def calculate_sec(power_kw, cycle_time_sec, production_rate_hr, scrap_rate):
    """
    Calculates Specific Energy Consumption (SEC): kWh per good unit produced.
    
    Formula:
        SEC = (Power_kW * (CycleTime_sec / 3600)) / (ProductionRate_hr * (1 - ScrapRate))
    """
    good_production_rate = max(0.1, float(production_rate_hr) * (1.0 - min(0.95, float(scrap_rate))))
    cycle_time_hr = float(cycle_time_sec) / 3600.0
    
    energy_per_cycle = float(power_kw) * cycle_time_hr
    sec = energy_per_cycle / (good_production_rate / 60.0) if good_production_rate > 0 else 999.0
    return round(float(sec), 4)

def calculate_pfc_simulation(p_kw, current_pf, target_pf=0.95):
    """
    Simulates Power Factor Correction (Capacitor Bank) to eliminate reactive power penalties.
    """
    current_pf = max(0.01, min(0.99, float(current_pf)))
    target_pf = max(current_pf, min(0.99, float(target_pf)))
    
    theta1 = np.arccos(current_pf)
    theta2 = np.arccos(target_pf)
    
    q1 = p_kw * np.tan(theta1)
    q2 = p_kw * np.tan(theta2)
    
    required_qc_kvar = max(0.0, q1 - q2)
    current_reduction_pct = max(0.0, (1.0 - (current_pf / target_pf)) * 100.0)
    
    # Estimated line I^2*R loss reduction (assumed ~3% distribution loss)
    distribution_loss_saving_kw = p_kw * 0.03 * (1.0 - (current_pf / target_pf)**2)
    
    return {
        "current_pf": round(current_pf, 3),
        "target_pf": round(target_pf, 3),
        "current_q_kvar": round(q1, 1),
        "target_q_kvar": round(q2, 1),
        "required_capacitor_bank_kvar": round(required_qc_kvar, 1),
        "line_current_reduction_pct": round(current_reduction_pct, 1),
        "distribution_loss_saving_kw": round(distribution_loss_saving_kw, 2)
    }

def train_energy_model(df):
    """
    Trains a surrogate Random Forest model to predict power consumption based on operating setpoints.
    Returns evaluation metrics.
    """
    X = df[ENERGY_FEATURES].fillna(0)
    y = df['Power_Consumption']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    joblib.dump(model, ENERGY_SURROGATE_PATH)
    
    importances = dict(zip(ENERGY_FEATURES, [round(float(v), 4) for v in model.feature_importances_]))
    
    return {
        "rmse": round(float(rmse), 3),
        "mae": round(float(mae), 3),
        "r2": round(float(r2), 4),
        "feature_importances": importances
    }

def load_energy_model(auto_retrain_on_error=True):
    """
    Loads pre-trained Energy Surrogate Model with fallback auto-training.
    """
    if os.path.exists(ENERGY_SURROGATE_PATH):
        try:
            return joblib.load(ENERGY_SURROGATE_PATH)
        except Exception as e:
            if not auto_retrain_on_error:
                raise e
    
    if auto_retrain_on_error:
        from src.data_loader import load_data
        sample_df = load_data()
        train_energy_model(sample_df)
        return joblib.load(ENERGY_SURROGATE_PATH)
        
    return None

def optimize_operating_setpoints(
    current_row,
    target_production_rate=None,
    electricity_cost_per_kwh=DEFAULT_ELECTRICITY_TARIFF,
    grid_emission_factor=DEFAULT_EMISSION_FACTOR,
    operating_hours_year=6000
):
    """
    Finds optimal (Shaft_Speed_RPM, Workload_Percentage) setpoints using SciPy's L-BFGS-B optimizer
    to minimize active power consumption and carbon footprint while satisfying production demands.
    """
    model = load_energy_model()
    if model is None:
        return None
    
    base_rpm = float(current_row['Shaft_Speed_RPM'])
    base_workload = float(current_row['Workload_Percentage'])
    base_power = float(current_row['Power_Consumption'])
    base_prod = float(current_row.get('Production_Rate', 30.0))
    
    if target_production_rate is None:
        target_production_rate = base_prod
        
    def objective(x):
        rpm_val, workload_val = x[0], x[1]
        
        # Physics affinity power law: P ~ (RPM/base_RPM)^2.0 and P ~ Workload
        power_ratio = (0.45 * (rpm_val / max(1.0, base_rpm))**2.0 + 
                       0.35 * (workload_val / max(1.0, base_workload)) + 
                       0.20)
        p_physics = base_power * power_ratio
        
        # Throughput preservation constraint penalty
        est_throughput = base_prod * (0.55 * (rpm_val / max(1.0, base_rpm)) + 0.45 * (workload_val / max(1.0, base_workload)))
        throughput_deficit = max(0.0, target_production_rate - est_throughput)
        
        penalty = 150.0 * (throughput_deficit ** 2)
        return p_physics + penalty

    # Bounds: RPM [800, 2600], Workload % [30, 95]
    bounds = [(800.0, 2600.0), (30.0, 95.0)]
    init_guess = [base_rpm, base_workload]
    
    res = minimize(objective, init_guess, method='L-BFGS-B', bounds=bounds)
    
    opt_rpm, opt_workload = float(res.x[0]), float(res.x[1])
    
    # Calculate optimized power
    opt_power_ratio = (0.45 * (opt_rpm / max(1.0, base_rpm))**2.0 + 
                       0.35 * (opt_workload / max(1.0, base_workload)) + 
                       0.20)
    opt_power = base_power * opt_power_ratio
    
    power_diff_kw = max(0.0, base_power - opt_power)
    savings_pct = (power_diff_kw / max(1.0, base_power)) * 100.0
    
    # Financial & Carbon calculations
    hourly_cost_saved = power_diff_kw * electricity_cost_per_kwh
    annual_cost_saved = power_diff_kw * electricity_cost_per_kwh * operating_hours_year
    annual_co2_saved_tons = (power_diff_kw * grid_emission_factor * operating_hours_year) / 1000.0
    
    # Baseline vs Optimized SEC
    cycle_time = float(current_row.get('Cycle_Time', 45.0))
    scrap_rate = float(current_row.get('Scrap_Rate', 0.02))
    baseline_sec = calculate_sec(base_power, cycle_time, base_prod, scrap_rate)
    optimized_sec = calculate_sec(opt_power, cycle_time, target_production_rate, scrap_rate)
    
    return {
        "base_rpm": round(base_rpm, 1),
        "opt_rpm": round(opt_rpm, 1),
        "delta_rpm": round(opt_rpm - base_rpm, 1),
        "base_workload": round(base_workload, 1),
        "opt_workload": round(opt_workload, 1),
        "delta_workload": round(opt_workload - base_workload, 1),
        "base_power_kw": round(base_power, 2),
        "opt_power_kw": round(opt_power, 2),
        "power_saved_kw": round(power_diff_kw, 2),
        "power_savings_percent": round(savings_pct, 1),
        "baseline_sec": round(baseline_sec, 4),
        "optimized_sec": round(optimized_sec, 4),
        "sec_reduction_percent": round(max(0.0, (baseline_sec - optimized_sec) / max(0.0001, baseline_sec) * 100.0), 1),
        "hourly_cost_saved_usd": round(hourly_cost_saved, 2),
        "annual_cost_saved_usd": round(annual_cost_saved, 2),
        "annual_co2_saved_metric_tons": round(annual_co2_saved_tons, 2)
    }
