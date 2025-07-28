# 🧠 Customer Lifetime Value (CLV) Predictor

Predict and optimize customer lifetime value using a fully automated, end-to-end machine learning pipeline. This solution empowers businesses to understand future revenue potential from their existing customer base, enabling strategic decision-making for retention, acquisition and personalized marketing.


## 📌 Objective

Customer Lifetime Value (CLV) is one of the most critical metrics in customer-centric businesses. This project aims to:

* **Accurately predict CLV** for individual customers using historical purchase behavior and customer attributes.
* **Enable real-time inference** via a scalable API and interactive dashboard.
---

## 💼 Business Impact

🔹 **Customer Segmentation** – Identify high-value customers and tailor offers.

🔹 **Churn Prevention** – Proactively retain customers projected to bring high value.

🔹 **Marketing ROI** – Allocate marketing spend effectively based on predicted CLV.

🔹 **Revenue Forecasting** – Better financial planning with forward-looking insights.

---

## 🚀 Tech Stack

| Category             | Tools Used                                          |
| -------------------- | --------------------------------------------------- |
| **Backend**          | Python, FastAPI                                     |
| **Dashboard**        | Streamlit                                           |
| **Database**         | Supabase PostgreSQL                                 |
| **ML Frameworks**    | scikit-learn, MLflow                                |
| **Containerization** | Docker, Docker Hub                                  |
| **Automation**       | GitHub Actions (CI/CD)                              |
| **Tracking**         | MLflow for experiment tracking and model versioning |
| **Deployment**       | Railway                                             |

---

## 🔄 Workflow Summary

1. **📦 Data Storage**
   Raw transactional and customer data stored in **Supabase PostgreSQL**.

2. **🔁 ETL Pipeline**

   * Extract → Transform → Load (ETL) pipeline written in Python.
   * Cleaned and structured data ready for training.

3. **🔍 Exploratory Data Analysis (EDA)**

   * Understanding purchase behavior, segment patterns, and outliers.

4. **⚙️ Feature Engineering & Preprocessing**

   * Numerical and categorical features processed using a modular, leakage-free pipeline.
   * Preprocessor saved and reused to ensure consistency in training and inference.

5. **📈 Model Training**

   * Trained a `RandomForestRegressor` using `RandomizedSearchCV` for hyperparameter tuning.
   * Evaluated using RMSE and R² scores.

6. **📊 MLflow Tracking**

   * Automatically logs:

     * Parameters
     * Evaluation metrics
     * Artifacts (model, preprocessor)

7. **📤 FastAPI App**

   * Exposes `/predict` endpoint to serve CLV predictions with confidence score.

8. **📊 Streamlit Dashboard**

   * Real-time interactive dashboard to predict CLV..

9. **🐳 Dockerization**

    * Both FastAPI and Streamlit apps are containerized for portability and consistency.
    * Images pushed to Docker Hub.

10. **🤖 CI/CD with GitHub Actions**

    * Automates build, test, and deploy workflows.
    * Deploys directly to Railway minimal manual effort.

---

## 🧪 How to Use

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/clv-predictor.git
   cd clv-predictor
   ```

2. **Run with Docker Compose**

   ```bash
   docker-compose up --build
   ```

3. **Access Applications**

   * FastAPI: [http://localhost:8000](http://localhost:8000)
   * Streamlit Dashboard: [http://localhost:8501](http://localhost:8501)

---

## 📦 Folder Structure

```
├── app/
│   ├── fastapi_app/            # FastAPI server logic
│   └── streamlit_app/          # Streamlit UI dashboard
|__ data                        # Folder to store data
|__ notebooks/eda               # Exploratory data analysis
├── src/
│   ├── pipeline/               # ETL pipeline
│   ├── model_train             # Training logic
|   |__model_predict
|   |__feature_engineering      # Adds the customer lifetime value column
|   |__monitoring/mlflow_helper # Mlflow for tracking and logging runs and metrics
│   ├── preprocessor.py         # Feature transformer
├── artifacts/                  # Saved model + preprocessor
├── docker-compose.yaml
├── Dockerfile.fastapi
├── Dockerfile.streamlit
├── .github/workflows/ci-cd pipeline          # GitHub Actions CI/CD
└── README.md
|__ requirements.txt
|__ run_feature_engineering
|__ run_model_predict
|__ run_model_train
|__ run_pipeline
|__ run_preprocessor
|__ setup
```

## 🛠 Features

✅ Real-Time CLV Prediction
✅ FastAPI + Streamlit Interface
✅ MLflow Integration for Full Experiment Tracking
✅ Docker-Ready and Production-Deployable
✅ Automated CI/CD with GitHub Actions


## 🧠 Author

**Sujato Dutta**
*Data Scientist | ML Engineer*
[LinkedIn](https://www.linkedin.com/in/sujato-dutta/)
[GitHub](https://github.com/Sujato-Dutta)
