"""
Model Training & Performance Evaluation Script for Siemens Process Energy Optimization.
"""
import os
import sys

# Ensure package directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data_loader import load_data
from src.energy_engine import train_energy_model

def main():
    print("================================================================")
    print("⚡ Siemens Energy Optimization Engine - ML Model Training")
    print("================================================================")
    df = load_data()
    print(f"Loaded {len(df):,} telemetry frames for training...")
    metrics = train_energy_model(df)
    
    print("\nTraining Run Completed Successfully!")
    print(f"• Surrogate Model R² Score: {metrics['r2']:.4f}")
    print(f"• Root Mean Squared Error (RMSE): {metrics['rmse']:.3f} kW")
    print(f"• Mean Absolute Error (MAE): {metrics['mae']:.3f} kW")
    print("\nTop Feature Importances for Power Demand:")
    for feat, imp in sorted(metrics['feature_importances'].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {feat:25s}: {imp*100:5.2f}%")
    print("================================================================")

if __name__ == "__main__":
    main()
