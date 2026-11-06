# Weather forecast prediction: SVM Kernel experiment vs Random Forest 🌧️🔮

An experimental machine learning benchmark study analyzing how different predictive architectures map independent, atmospheric features to predict rainfall. This project evaluates tree-based ensemble partitioning against high-dimensional geometric decision boundaries.

## 🎯 Project overview & goal
The core goal of this experiment is to isolate how changing an algorithm's mathematical engine impacts its ability to map non-correlated weather metrics. 

A **Random Forest Classifier** is evaluated alongside a **Support Vector Machine (SVM)** deployed across two distinct hyperparameter configurations: a rigid **Linear Kernel** and a higher-dimensional **Radial Basis Function (RBF) Kernel**.

## 📊 Dataset profile
The project utilizes a 2,500-row weather forecast dataset tracking key meteorological variables:
*   **Features:** Temperature, Humidity, Wind Speed, Cloud Cover, Pressure
*   **Target:** `Rain` (Binary: 1 = Rain, 0 = No Rain)

### 🔬 Rigorous Data Diagnostics (EDA)
*   **No data leakage:** A systematic duplicate check confirmed **0 duplicate rows**, verifying that model accuracies reflect true predictive learning rather than row memorization.
*   **Completely decoupled Features:** Pearson correlation matrix testing revealed extremely low feature-to-feature coefficients (ranging between **$\pm0.01$ and $\pm0.03$**). This confirms a synthetic data environment where features lack natural linear thermodynamic ties.
*   **Class imbalance mitigation:** The target variable exhibits a heavy **82/18 class split** (clear days heavily outnumbering rainy days). To prevent training bias, a stratified split (`stratify=y`) was enforced to lock identical class distributions into both subsets.

---

## 🛠️ Machine learning pipeline

1.  **Text mapping:** Transformed the string target values (`rain` / `no rain`) into binary numeric structures (`1` / `0`) using text cleaning (`.str.strip().str.lower()`).
2.  **Stratified split:** Executed an 80/20 train-test split utilizing stratification on the target variable.
3.  **Feature normalization:** Applied a `StandardScaler` to ensure the continuous weather variables were uniformly scaled—a mandatory step to prevent geometric distortion in the SVM hyperplane.
4.  **Benchmarking:** Trained and evaluated Random Forest, Linear SVM, and RBF SVM models simultaneously.

---

## 🏆 Experimental results & insights

### Evaluation metrics

| Algorithm / Architecture | Overall Accuracy | Rain Class (1) F1-Score | Data Requirement |
| :--- | :---: | :---: | :--- |
| **1. Random Forest** | **100%** | **1.00** | Raw or Scaled |
| **2. Linear SVM** | **93%** | **0.70** | Must be Scaled |
| **3. RBF SVM** | **97%** | **0.90** | Must be Scaled |

### Key takeaways
*   **The Random Forest rule:** The Random Forest achieved a flawless 1.00 baseline. Because it creates sharp, perpendicular splits, it cleanly reverse-engineered the underlying conditional threshold rules used to generate this synthetic data.
*   **The Linear hyperplane penalty:** The Linear SVM struggled significantly on the minority class ($0.70$ F1-Score). It missed **20 rainy days entirely (False Negatives)**. Because the weather features share zero linear correlation, a flat line is mathematically blind to the data pockets.
*   **The Power of the Kernel trick:** The RBF SVM drastically recovered the geometric boundary. By warping the uncorrelated data points into a higher geometric dimension, it successfully dropped missed-rain errors **from 20 down to just 6**, proving the necessity of non-linear vector mapping.

