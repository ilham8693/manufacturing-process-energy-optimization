"""
Siemens Process Energy Optimization & Carbon Reduction Engine
Enterprise SCADA Digital Twin & SciPy Constrained Setpoint Optimizer
"""
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="Siemens Energy Optimization & Carbon Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure local package imports work seamlessly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_data
from src.energy_engine import (
    calculate_power_quality_metrics,
    calculate_sec,
    calculate_pfc_simulation,
    optimize_operating_setpoints,
    DEFAULT_EMISSION_FACTOR,
    DEFAULT_ELECTRICITY_TARIFF
)

# Custom High-Contrast SCADA Styling
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #0b0f17;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Header Banner */
    .dashboard-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-bottom: 2px solid #0284c7;
        padding: 22px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .header-title {
        color: #ffffff;
        font-size: clamp(1.4rem, 2.2vw, 2.0rem);
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
        font-weight: 400;
    }

    /* SCADA KPI Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #161f30;
        border: 1px solid #22324d;
        border-radius: 10px;
        padding: 16px 18px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
    }
    .kpi-card.blue::before { background: #0284c7; }
    .kpi-card.green::before { background: #10b981; }
    .kpi-card.amber::before { background: #f59e0b; }
    .kpi-card.purple::before { background: #8b5cf6; }
    .kpi-card.red::before { background: #ef4444; }

    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: clamp(1.3rem, 1.8vw, 1.8rem);
        font-weight: 700;
        color: #ffffff;
        line-height: 1.1;
        margin-bottom: 6px;
        font-variant-numeric: tabular-nums;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #cbd5e1;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-optimal { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #10b981; }
    .badge-warning { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #f59e0b; }
    .badge-alert { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #ef4444; }

    /* Section Cards */
    .section-card {
        background: #131b2a;
        border: 1px solid #1e2c44;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .section-title {
        color: #f1f5f9;
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 14px;
        border-bottom: 1px solid #22324d;
        padding-bottom: 8px;
    }
    
    /* Highlight Box */
    .savings-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(2, 132, 199, 0.12) 100%);
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_cached_telemetry():
    return load_data()

df_all = get_cached_telemetry()

# ==========================================
# SIDEBAR CONTROLS & OPERATIONAL PARAMETERS
# ==========================================
st.sidebar.markdown("### ⚙️ Machine & Operational Scope")
machine_list = sorted(df_all['Machine_ID'].unique().tolist())
selected_machine = st.sidebar.selectbox("Industrial Machine ID", machine_list, index=0)
df_mach = df_all[df_all['Machine_ID'] == selected_machine].reset_index(drop=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌱 Economic & Grid Parameters")
electricity_cost = st.sidebar.number_input(
    "Electricity Tariff ($ / kWh)",
    min_value=0.01,
    max_value=1.00,
    value=DEFAULT_ELECTRICITY_TARIFF,
    step=0.01,
    help="Utility commercial electricity tariff per kilowatt-hour"
)

emission_factor = st.sidebar.number_input(
    "Grid Emission Factor (kg CO₂ / kWh)",
    min_value=0.05,
    max_value=1.50,
    value=DEFAULT_EMISSION_FACTOR,
    step=0.05,
    help="Regional electric grid carbon intensity factor (US/EU industrial avg: ~0.45 kg CO2/kWh)"
)

operating_hours_yr = st.sidebar.number_input(
    "Annual Operating Hours",
    min_value=500,
    max_value=8760,
    value=6000,
    step=500,
    help="Annual machine utilization hours (standard 3-shift factory: ~6,000 hrs/year)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ Telemetry Timeline Scrubber")
max_idx = min(1000, len(df_mach) - 1)
frame_idx = st.sidebar.slider(
    "Replay Frame (1-Min Intervals)",
    min_value=0,
    max_value=max_idx,
    value=min(120, max_idx),
    help="Scrub across 1-minute SCADA intervals to analyze instantaneous 3-phase electrical and thermal states."
)

current_record = df_mach.iloc[frame_idx]
record_ts = current_record['Timestamp']
if isinstance(record_ts, pd.Timestamp):
    formatted_ts = record_ts.strftime('%Y-%m-%d %H:%M:%S')
else:
    formatted_ts = str(record_ts)

st.sidebar.info(f"📅 **Timestamp:** `{formatted_ts}`\n\n📌 **Frame Index:** `{frame_idx} of {max_idx}`")

# ==========================================
# CALCULATE KEY METRICS FOR CURRENT FRAME
# ==========================================
pq = calculate_power_quality_metrics(current_record)
sec_val = calculate_sec(
    current_record['Power_Consumption'],
    current_record['Cycle_Time'],
    current_record['Production_Rate'],
    current_record['Scrap_Rate']
)
co2_rate_kg_hr = current_record['Power_Consumption'] * emission_factor
cost_rate_usd_hr = current_record['Power_Consumption'] * electricity_cost

# ==========================================
# HEADER SECTION
# ==========================================
st.markdown(f"""
<div class="dashboard-header">
    <div class="header-title">
        <span>⚡ Siemens Process Energy Optimization & Carbon Reduction Engine</span>
    </div>
    <div class="header-subtitle">
        Real-Time 3-Phase Power Quality, Thermodynamic Heat Balances, Specific Energy (SEC) & SciPy Setpoint Optimizer | 
        <strong>Machine: {selected_machine}</strong> | 📅 <strong>Telemetry Time: {formatted_ts}</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SCADA EXECUTIVE METRIC CARDS
# ==========================================
vuf_badge = "badge-optimal" if pq['VUF_percent'] < 2.0 else "badge-warning"
iuf_badge = "badge-optimal" if pq['IUF_percent'] < 5.0 else "badge-warning"

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card blue">
        <div class="kpi-label">Active Power Demand</div>
        <div class="kpi-value">{current_record['Power_Consumption']:.1f} <span style="font-size: 0.9rem; font-weight: 400; color: #94a3b8;">kW</span></div>
        <div class="kpi-sub">
            <span>PF: <strong>{pq['Power_Factor']:.3f}</strong></span> • <span>Loss: <strong>{pq['Thermal_Loss_kW']:.1f} kW</strong></span>
        </div>
    </div>
    <div class="kpi-card green">
        <div class="kpi-label">Specific Energy (SEC)</div>
        <div class="kpi-value">{sec_val:.3f} <span style="font-size: 0.9rem; font-weight: 400; color: #94a3b8;">kWh/part</span></div>
        <div class="kpi-sub">
            <span>Efficiency: <strong>{current_record['Energy_Efficiency_Index']:.1f}%</strong></span>
        </div>
    </div>
    <div class="kpi-card amber">
        <div class="kpi-label">3-Phase Voltage Unbalance (VUF)</div>
        <div class="kpi-value">{pq['VUF_percent']:.2f}%</div>
        <div class="kpi-sub">
            <span class="badge {vuf_badge}">{pq['VUF_status']}</span>
        </div>
    </div>
    <div class="kpi-card red">
        <div class="kpi-label">Carbon Emission Rate</div>
        <div class="kpi-value">{co2_rate_kg_hr:.1f} <span style="font-size: 0.9rem; font-weight: 400; color: #94a3b8;">kg CO₂/h</span></div>
        <div class="kpi-sub">
            <span>Cost: <strong>${cost_rate_usd_hr:.2f} / hr</strong></span>
        </div>
    </div>
    <div class="kpi-card purple">
        <div class="kpi-label">Reactive Power (Q)</div>
        <div class="kpi-value">{pq['Reactive_Power_kVAR']:.1f} <span style="font-size: 0.9rem; font-weight: 400; color: #94a3b8;">kVAR</span></div>
        <div class="kpi-sub">
            <span>Apparent (S): <strong>{pq['Apparent_Power_kVA']:.1f} kVA</strong></span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MAIN INTERACTIVE TABS
# ==========================================
tab_opt, tab_pq, tab_historical, tab_pfc = st.tabs([
    "🎯 AI Setpoint Optimizer (SciPy)",
    "⚡ 3-Phase Power Quality & Thermal Balance",
    "📊 Historical Efficiency & Operating Regimes",
    "🔌 Power Factor Correction (Capacitor Bank)"
])

# -------------------------------------------------------------
# TAB 1: AI SETPOINT OPTIMIZER (SCIPY CONSTRAINED OPTIMIZATION)
# -------------------------------------------------------------
with tab_opt:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🎯 Constrained Non-Linear Setpoint Optimization (L-BFGS-B Engine)</div>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 16px;">
            This optimization engine computes the thermodynamic & electrical Pareto frontier between <strong>Spindle Speed (RPM)</strong> 
            and <strong>Workload (%)</strong> to minimize active power consumption and specific energy consumption (SEC), 
            subject to guaranteed production throughput and machine physical operating limits.
        </p>
    """, unsafe_allow_html=True)
    
    col_input, col_action = st.columns([2, 1])
    
    with col_input:
        curr_prod = float(current_record['Production_Rate'])
        target_prod = st.slider(
            "Target Production Throughput (parts / hour)",
            min_value=max(5.0, curr_prod * 0.7),
            max_value=max(35.0, curr_prod * 1.3),
            value=curr_prod,
            step=0.5,
            help="Required factory output. The optimizer will enforce this constraint while trimming excessive speed & load power losses."
        )
    
    with col_action:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_opt = st.button("⚡ Run Real-Time Setpoint Optimization", type="primary", use_container_width=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

    # Perform Optimization
    opt_res = optimize_operating_setpoints(
        current_record,
        target_production_rate=target_prod,
        electricity_cost_per_kwh=electricity_cost,
        grid_emission_factor=emission_factor,
        operating_hours_year=operating_hours_yr
    )

    if opt_res:
        st.markdown("### 📋 Optimization Results: Baseline vs. Optimized State")
        
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            st.metric(
                label="Shaft Speed (RPM)",
                value=f"{opt_res['opt_rpm']:.0f} RPM",
                delta=f"{opt_res['delta_rpm']:+.0f} RPM",
                delta_color="inverse" if opt_res['delta_rpm'] > 0 else "normal"
            )
        with c_m2:
            st.metric(
                label="Workload Percentage",
                value=f"{opt_res['opt_workload']:.1f}%",
                delta=f"{opt_res['delta_workload']:+.1f}%",
                delta_color="off"
            )
        with c_m3:
            st.metric(
                label="Active Power Demand",
                value=f"{opt_res['opt_power_kw']:.1f} kW",
                delta=f"-{opt_res['power_saved_kw']:.1f} kW ({opt_res['power_savings_percent']}%)",
                delta_color="normal"
            )
        with c_m4:
            st.metric(
                label="Specific Energy (SEC)",
                value=f"{opt_res['optimized_sec']:.3f} kWh/unit",
                delta=f"-{opt_res['sec_reduction_percent']}% SEC",
                delta_color="normal"
            )

        # Financial & Carbon Savings Ledger
        st.markdown(f"""
        <div class="savings-banner">
            <h4 style="margin: 0 0 10px 0; color: #10b981; font-weight: 700; font-size: 1.1rem;">
                🌱 Estimated Economic & Decarbonization Impact (Based on {operating_hours_yr:,} Operating Hours/Year):
            </h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;">
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem;">Hourly Energy Reduction:</span><br>
                    <strong style="color: #ffffff; font-size: 1.15rem;">{opt_res['power_saved_kw']:.2f} kWh / hr</strong>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem;">Annual Cost Savings:</span><br>
                    <strong style="color: #34d399; font-size: 1.15rem;">${opt_res['annual_cost_saved_usd']:,.2f} USD / yr</strong>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem;">Carbon Footprint Mitigated:</span><br>
                    <strong style="color: #38bdf8; font-size: 1.15rem;">{opt_res['annual_co2_saved_metric_tons']:.2f} Metric Tons CO₂ / yr</strong>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem;">Equivalent Trees Planted:</span><br>
                    <strong style="color: #fbbf24; font-size: 1.15rem;">~{int(opt_res['annual_co2_saved_metric_tons'] * 45):,} Mature Trees / yr</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparison Bar Chart
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Baseline Telemetry",
            x=["Power (kW)", "Shaft Speed (/10 RPM)", "Workload (%)", "SEC (x100 kWh/u)"],
            y=[opt_res['base_power_kw'], opt_res['base_rpm']/10.0, opt_res['base_workload'], opt_res['baseline_sec']*100],
            marker_color="#ef4444"
        ))
        fig_comp.add_trace(go.Bar(
            name="SciPy Optimized Setpoint",
            x=["Power (kW)", "Shaft Speed (/10 RPM)", "Workload (%)", "SEC (x100 kWh/u)"],
            y=[opt_res['opt_power_kw'], opt_res['opt_rpm']/10.0, opt_res['opt_workload'], opt_res['optimized_sec']*100],
            marker_color="#10b981"
        ))
        fig_comp.update_layout(
            title="Baseline Operating Telemetry vs. SciPy Optimal Setpoint",
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="#131b2a",
            plot_bgcolor="#131b2a",
            height=340,
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: 3-PHASE POWER QUALITY & THERMAL BALANCE
# -------------------------------------------------------------
with tab_pq:
    st.markdown("### ⚡ 3-Phase Electrical Unbalance & Power Triangle")
    col_pq1, col_pq2 = st.columns([1, 1])
    
    with col_pq1:
        v_df = pd.DataFrame({
            "Phase": ["Phase A", "Phase B", "Phase C"],
            "Voltage (V)": [pq['V_Phase_A'], pq['V_Phase_B'], pq['V_Phase_C']],
            "Current (A)": [pq['I_Phase_A'], pq['I_Phase_B'], pq['I_Phase_C']]
        })
        
        fig_phases = go.Figure()
        fig_phases.add_trace(go.Bar(
            x=v_df['Phase'], y=v_df['Voltage (V)'], name="Phase Voltage (V)", marker_color="#0284c7"
        ))
        fig_phases.add_trace(go.Bar(
            x=v_df['Phase'], y=v_df['Current (A)'], name="Phase Current (A)", marker_color="#f59e0b", yaxis="y2"
        ))
        fig_phases.update_layout(
            title=f"3-Phase Voltage & Current Balance (VUF: {pq['VUF_percent']:.2f}% | IUF: {pq['IUF_percent']:.2f}%)",
            template="plotly_dark",
            paper_bgcolor="#131b2a",
            plot_bgcolor="#131b2a",
            height=340,
            yaxis=dict(title="Voltage (V)", range=[0, max(260, max(v_df['Voltage (V)']) * 1.15)]),
            yaxis2=dict(title="Current (A)", overlaying="y", side="right", range=[0, max(150, max(v_df['Current (A)']) * 1.25)]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_phases, use_container_width=True)

    with col_pq2:
        # Power Triangle Sankey/Donut
        fig_power = go.Figure(data=[go.Pie(
            labels=["Active Useful Power (kW)", "Motor Thermal Losses (kW)", "Reactive Demand (kVAR)"],
            values=[
                max(0.1, pq['Active_Power_kW'] - pq['Thermal_Loss_kW']),
                max(0.1, pq['Thermal_Loss_kW']),
                max(0.1, pq['Reactive_Power_kVAR'])
            ],
            hole=0.55,
            marker=dict(colors=["#10b981", "#ef4444", "#8b5cf6"])
        )])
        fig_power.update_layout(
            title="Apparent Power Decomposition (Useful Work vs Loss vs Reactive)",
            template="plotly_dark",
            paper_bgcolor="#131b2a",
            height=340,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_power, use_container_width=True)

    # Thermal Gradient & Heat Balance
    st.markdown("### 🌡️ Thermal Equilibrium & Component Heat Dissipation")
    c_th1, c_th2, c_th3, c_th4 = st.columns(4)
    with c_th1:
        st.metric("Motor Temperature", f"{current_record['Motor_Temperature']:.1f} °C", f"Limit: 95°C")
    with c_th2:
        st.metric("Bearing Temperature", f"{current_record['Bearing_Temperature']:.1f} °C", f"Limit: 85°C")
    with c_th3:
        delta_t = current_record['Motor_Temperature'] - current_record['Bearing_Temperature']
        st.metric("Thermal Gradient (ΔT)", f"{delta_t:+.1f} °C", "Motor - Bearing")
    with c_th4:
        st.metric("Coolant Flow Rate", f"{current_record['Coolant_Flow_Rate']:.1f} L/min", f"Temp: {current_record['Coolant_Temperature']:.1f}°C")

# -------------------------------------------------------------
# TAB 3: HISTORICAL EFFICIENCY & OPERATING REGIMES
# -------------------------------------------------------------
with tab_historical:
    st.markdown("### 📊 Historical Machine Operating Regimes & Specific Energy Mapping")
    
    sample_hist = df_mach.iloc[:1000]
    fig_scatter = px.scatter(
        sample_hist,
        x="Shaft_Speed_RPM",
        y="Power_Consumption",
        color="Energy_Efficiency_Index",
        size="Workload_Percentage",
        hover_data=["Cycle_Time", "Production_Rate", "Scrap_Rate", "RMS_Vibration"],
        title=f"Power Consumption (kW) vs. Spindle Speed (RPM) across {len(sample_hist)} Telemetry Intervals",
        template="plotly_dark",
        height=450,
        color_continuous_scale="Turbo"
    )
    fig_scatter.update_layout(
        paper_bgcolor="#131b2a",
        plot_bgcolor="#131b2a",
        margin=dict(l=40, r=20, t=50, b=40)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: POWER FACTOR CORRECTION (CAPACITOR BANK SIMULATION)
# -------------------------------------------------------------
with tab_pfc:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🔌 Power Factor Correction (PFC) & Reactive Relief Simulator</div>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 14px;">
            Low power factor induces heavy reactive currents in distribution feeders, leading to utility kVAR penalties and $I^2 R$ heat dissipation.
            Simulate the installation of a power capacitor bank to achieve unity/optimal power factor.
        </p>
    """, unsafe_allow_html=True)
    
    col_pfc_in, col_pfc_out = st.columns([1, 1])
    with col_pfc_in:
        current_pf = pq['Power_Factor']
        st.write(f"Current Measured Power Factor: **{current_pf:.3f}**")
        target_pf = st.slider("Target Power Factor (PF)", min_value=max(0.85, current_pf), max_value=0.99, value=0.95, step=0.01)
        
        pfc_sim = calculate_pfc_simulation(current_record['Power_Consumption'], current_pf, target_pf=target_pf)
        
    with col_pfc_out:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Required Capacitor Bank", f"{pfc_sim['required_capacitor_bank_kvar']:.1f} kVAR")
        with m2:
            st.metric("Feeder Line Current Relief", f"-{pfc_sim['line_current_reduction_pct']:.1f}%")
        
        st.info(f"💡 **Distribution Gain:** Reduces line $I^2R$ copper losses by **{pfc_sim['distribution_loss_saving_kw']:.2f} kW** "
                f"and eliminates utility reactive demand surcharges.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# DATASET PROVENANCE & METHODOLOGY FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 12px 0;">
    ⚡ <strong>Siemens Process Energy Optimization Engine</strong> | 
    Dataset sourced from <a href="https://www.kaggle.com/" target="_blank" style="color: #38bdf8; text-decoration: none;">Kaggle.com</a> 
    (Siemens Industrial Predictive Maintenance & Energy Telemetry).<br>
    Formulations adhere to IEEE 141 (Voltage Unbalance) & Constrained SciPy L-BFGS-B Optimization.
</div>
""", unsafe_allow_html=True)
