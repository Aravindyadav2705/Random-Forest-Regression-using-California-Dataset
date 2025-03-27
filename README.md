# Random Forest Regression on the California Housing Dataset

This repository demonstrates a complete workflow for **regression** using a **Random Forest** to predict median house values in California. The project covers data loading, splitting, model training, evaluation, and a range of visualizations to ensure clarity, reproducibility, and interpretability.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features & Visualizations](#features--visualizations)  
3. [Prerequisites & Installation](#prerequisites--installation)  
4. [Usage](#usage)  
5. [Repository Structure](#repository-structure)  
6. [Detailed Explanation](#detailed-explanation)  
7. [Results](#results)  
8. [Accessibility](#accessibility)  
9. [License](#license)  
10. [Clone the Repository](#clone-the-repository)  

---

## Project Overview

The **California Housing** dataset, originally sourced from the 1990 U.S. Census, is a well-known benchmark for housing price prediction. It includes **8 features** such as median income, average occupancy, and geographical coordinates (latitude, longitude). The target variable, **MedHouseVal**, represents the median house value in each block group (in hundreds of thousands of dollars).

By training a **Random Forest Regressor** on this dataset, we aim to:

- Predict house values with high accuracy.
- Evaluate model performance using R² and Mean Squared Error.
- Visualize how the model’s predictions align with actual data.
- Identify the most influential features using feature importance.
- Explore partial dependence for a deeper understanding of how specific features (e.g., median income) affect predicted house value.

---

## Features & Visualizations

1. **Data Preprocessing**  
   - The dataset is loaded via `fetch_california_housing(as_frame=True)` into a pandas DataFrame.
   - The target variable is separated from the input features for clarity.

2. **Random Forest Model**  
   - A `RandomForestRegressor(n_estimators=100, random_state=42)` is trained on 70% of the data.
   - The model leverages bootstrap sampling and ensemble averaging to reduce variance.

3. **Performance Metrics**  
   - **R² Score**: Measures how much variance in the target is explained by the model.  
   - **Mean Squared Error (MSE)**: Averages the squared differences between predictions and actual values.

4. **Rich Visualizations**  
   - **Scatter Plot (Actual vs. Predicted)**: Depicts alignment of predictions with true house values.  
   - **Residual Plot**: Reveals potential patterns or biases in the model’s errors.  
   - **Feature Importance Bar Chart**: Highlights which features contribute most to the model’s decisions.  
   - **Partial Dependence Plot**: Shows how a single feature (e.g., median income) influences predicted house value.

---

## Prerequisites & Installation

- **Python 3.7+**
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)


## Data Loading
We use `fetch_california_housing(as_frame=True)` to load the dataset into a DataFrame. The target (MedHouseVal) is separated from the features for a clear regression setup.

## Model Training
A Random Forest Regressor with 100 estimators is trained on 70% of the data.  
The model captures complex, non-linear relationships through an ensemble of decision trees.

## Evaluation
- **R² Score:** Gauges how much of the variance in median house value is explained by the model.
- **Mean Squared Error:** Quantifies the average squared deviation between predictions and actual values.

## Visualization
- **Actual vs. Predicted:** Verifies overall alignment between the model’s predictions and ground truth.
- **Residual Plot:** Diagnoses potential systematic errors.
- **Feature Importance:** Ranks features by their contribution to reducing prediction error.
- **Partial Dependence:** Illustrates how changes in a single feature (e.g., median income) influence predictions, providing insight into the model’s behavior.

## Results
Typical performance (may vary slightly depending on system/environment):
- **R² Score:** ~0.80 (indicating ~80% of variance explained)
- **Mean Squared Error:** ~0.25 (on the scale of hundreds of thousands of dollars)

**Visualizations:**
- Strong clustering along the diagonal in the Actual vs. Predicted plot.
- Residuals near zero for most predictions, but some underestimation at the high-value range.
- **MedInc** typically emerges as the top feature, followed by **Latitude**, **Longitude**, etc.
- Partial Dependence for **MedInc** reveals a clear upward trend with higher incomes.

## Accessibility
- **Color Schemes:** Uses colorblind-friendly palettes (e.g., “seaborn-darkgrid”) and distinct color markers.
- **Clear Labeling:** All figures include descriptive titles, axis labels, and legends.
- **Screen Reader Compatibility:** Captions and alt-text (if provided) ensure interpretability for screen readers.
- **Transcripts/Closed Captions:** Recommended if any multimedia content accompanies the tutorial.

## License
This project is distributed under the **MIT License**. You are free to use, modify, and distribute the code for personal or commercial purposes, provided you include the original license text.

## Clone the Repository
```bash
git clone https://github.com/Aravindyadav2705/RandomForest-CaliforniaHousing.git
