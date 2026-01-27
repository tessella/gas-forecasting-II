### UK Gas Demand Forecasting II: 

- **Core Method:** Adapted my own methodology (in turn adapted from [Transportation Marketplace Rate Forecast Using Signature Transform](https://pubsonline.informs.org/doi/10.1287/inte.2025.0251)) to model the physical dynamics of UK natural gas demand.
- **Implementation Challenges:** My original "local normalization" (Z-scoring) failed for gas forecasting, where absolute temperature levels drive demand. Additionally, the linear model could not capture the non-linearity of thermal efficiency in extreme cold.
- **Key Modifications:**
    - **Physics-Aware Normalization:** Replaced window-based scaling with global scaling to preserve absolute thermal signals (global scaling is acceptable for weather data as it is considered stationary).
    - **Global-Soft Weighting:** Pivoted to a globally-weighted Ridge Regression (Temperature $\approx$ 0.1) to utilize the full history while retaining soft regime-switching for structural breaks.
    - **Capturing Non-Linearity:** Computed level 3 signature on features to capture convexity in the data.
- **Results:** Achieved a **5.96% MAPE** (7-day ahead) and **7.92% MAPE** (14-day ahead), outperforming standard industry benchmarks of [10%](https://www.mdpi.com/1996-1073/14/16/4905) and competing with commercial utility-grade models with a simple dataset.
