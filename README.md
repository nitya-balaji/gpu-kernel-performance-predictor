# GPU Kernel Performance Predictor
**End-to-End Machine Learning System (Production-Ready)**

This project is a full end-to-end Machine Learning pipeline designed to predict GPU kernel runtimes based on OpenCL SGEMM configuration parameters. It demonstrates real-world ML engineering practices, including modular code design, custom logging, exception handling, CI/CD automation, Dockerization, and cloud deployment on AWS.

---

## Problem Statement

GPU performance is highly sensitive to kernel configuration parameters such as work-group sizes, vector widths, and memory access patterns. Selecting the wrong configuration can lead to significantly slower runtimes, making performance tuning a costly trial-and-error process.

This project uses the following dataset for training: [GPU Kernel Performance Dataset — Kaggle](https://www.kaggle.com/datasets/rupals/gpu-runtime?select=sgemm_product.csv)

The dataset contains 14 OpenCL SGEMM tuning parameters (`MWG`, `NWG`, `KWG`, `MDIMC`, `NDIMC`, `MDIMA`, `NDIMB`, `KWI`, `VWM`, `VWN`, `STRM`, `STRN`, `SA`, `SB`) and a target variable `Runtime` (in milliseconds). The objective is to predict GPU kernel execution time given a configuration, enabling developers to identify optimal kernel settings without exhaustive benchmarking.

---

## Solution Overview

### Exploratory Data Analysis (`notebook/eda.ipynb`)

Before training any model, a thorough analysis of the dataset was conducted to understand the underlying structure of GPU kernel performance:

- **Feature Profiling** — Examined each of the 14 GPU architecture parameters (work-group sizes, vector widths, memory access flags) against the target variable `Runtime` to understand their individual impact on execution time
- **Correlation Analysis** — Identified non-linear relationships between parameters such as `MDIMC`, `NDIMC`, and `Runtime`, revealing that simple linear models would be insufficient for this dataset
- **Outlier Detection** — Flagged configurations that caused abnormal slowdowns to prevent the model from learning from noisy, unrepresentative data points
- **Visualization** — Used scatter plots and heatmaps to identify which kernel tuning parameters most strongly influence GPU execution speed

### Model Training (`notebook/model_training.ipynb`)

The notebook served as the experimental ground for preprocessing and model selection before the production pipeline was built:

- **Preprocessing** — Applied `StandardScaler` to normalize all 14 numerical features, accounting for the wide range of parameter values across different kernel configurations
- **Model Selection** — Evaluated multiple regression approaches suited to the non-linear complexity of GPU performance data, including Random Forest, XGBoost, CatBoost, Gradient Boosting, Decision Tree, Linear Regression, and AdaBoost
- **Evaluation** — Measured model performance using R² score to assess how closely predictions matched actual measured runtimes

### Production ML Pipeline

- **Data Ingestion (`src/components/data_ingestion.py`)**
  - Automated loading of raw CSV data from `notebook/data/sgemm_product_v2.csv`
  - 80/20 train-test split saved to `artifacts/` for use in data transformation and model training

- **Data Transformation (`src/components/data_transformation.py`)**
  - Standardized all 14 numerical input features using a preprocessing pipeline
  - Saved the fitted preprocessor as `artifacts/preprocessor.pkl` to ensure identical scaling rules are applied when making predictions

- **Model Training (`src/components/model_trainer.py`)**
  - Trained 7 regression models with `GridSearchCV` hyperparameter tuning and 2-fold cross-validation
  - Selected the best performing model based on R² score on the held-out test set (with final R² Score: 0.9999) 
  - Saved the best model as `artifacts/model.pkl`

- **Training Pipeline (`src/pipeline/train_pipeline.py`)**
  - Orchestrates all 3 components above in sequence
  - Triggered on demand via the `/train` route in the Flask app

- **Prediction Pipeline (`src/pipeline/predict_pipeline.py`)**
  - Separate prediction pipeline for real-time predictions
  - Loads `model.pkl` and `preprocessor.pkl` to ensure consistent preprocessing between training and prediction
  - Accepts user input from the Flask web app and returns a predicted runtime in milliseconds

### Model Deployment
- Exposed the trained model via a **Flask** web application
- Containerized the application using **Docker**
- Pushed the Docker image to **Amazon ECR (Elastic Container Registry)**
- Deployed the container to **AWS EC2** (Ubuntu 24.04, t3.small, 20GB storage)
- Automated the full build and deployment cycle using **GitHub Actions CI/CD** with a self-hosted runner on EC2

> ⚠️ **Note on Cloud Deployment** - To avoid ongoing cloud charges:
> - The EC2 instance has been deleted
> - The self-hosted GitHub Actions runner has been removed
>
> The project remains fully reproducible locally using Docker and the modular ML pipelines below.

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/nitya-balaji/gpu-kernel-performance-predictor
cd gpu-kernel-performance-predictor
```

**2. Build the Docker image**
```bash
docker build -t gpu-performance-predictor .
```

**3. Run the container**
```bash
docker run -p 5000:5000 gpu-performance-predictor
```

**4. Train the model**

Open your browser and visit:
```
http://localhost:5000/train
```
Wait for the page to display:
```
Training complete! R2 Score: 0.9999...
```
⚠️ This step trains all 7 models and selects the best one — it may take several minutes to complete. Do not close the browser tab until training is finished. This must be completed before making any predictions.

**5. Navigate to:**
```
http://localhost:5000/predictdata
```
Enter your GPU kernel configuration parameters and click **Predict Runtime** to get a predicted runtime in milliseconds.

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/main.yaml`) automates the following on every push to `main`:

1. **Continuous Integration** — Lint and unit test checks run on `ubuntu-latest`
2. **Continuous Delivery** — Docker image is built and pushed to Amazon ECR
3. **Continuous Deployment** — EC2 self-hosted runner pulls the latest image from ECR and runs the updated container on port 5000

---

## Web App Preview
<img src="assets/homepage.png" width="600"/>
<img src="assets/predict2.png" width="600"/>
<img src="assets/runtime.png" width="600"/>

