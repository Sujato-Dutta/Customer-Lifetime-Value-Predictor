import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model and preprocessor
model = joblib.load("artifacts/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

class InputData(BaseModel):
    purchase_amount: float
    product_category: str
    payment_method: str
    customer_segment: str
    region: str

@app.post("/predict")
def predict(data: InputData):
    input_dict = data.model_dump()

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])  # wrap in list create single-row DataFrame
    # Preprocess
    input_processed = preprocessor.transform(input_df)
    # Predict
    prediction = model.predict(input_processed)
    return {"predicted_clv": float(prediction[0])}
