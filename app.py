from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# ---------------- LOAD MODELS ---------------- #

binary_model = joblib.load("binary_model.pkl")
multiclass_model = joblib.load("multiclass_model.pkl")
multilabel_model = joblib.load("multilabel_model.pkl")  # cc_model

le = joblib.load("label_encoder.pkl")
mlb = joblib.load("mlb.pkl")
imputer_ml = joblib.load("imputer_ml.pkl")
freq_maps = joblib.load("freq_maps.pkl")
# ---------------- EXPECTED FEATURES ---------------- #

EXPECTED_COLUMNS = [
    "Clinic","Service","AmountCharged","CPTCode","Payer","Provider",
    "BillingProviderNPI","ClaimFacilityNPI","AuthStatus","eligStatus",
    "CoPay","Deduc","CoIns","SameDayCli","DaysBetServiceToBilling",
    "tpcliStrModifier","tpcliStrPOS","f21diag1",
    "service_year","service_month","billing_year","billing_month","patient_age"
]

# ---------------- PREPROCESS ---------------- #

def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    # -------- DATE PROCESSING -------- #
    df["ServiceDt"] = pd.to_datetime(df["ServiceDt"])
    df["ClaimBillDate"] = pd.to_datetime(df["ClaimBillDate"])
    df["f11insdob"] = pd.to_datetime(df["f11insdob"])

    df["service_year"] = df["ServiceDt"].dt.year
    df["service_month"] = df["ServiceDt"].dt.month

    df["billing_year"] = df["ClaimBillDate"].dt.year
    df["billing_month"] = df["ClaimBillDate"].dt.month

    df["patient_age"] = (df["ServiceDt"] - df["f11insdob"]).dt.days / 365

    df = df.drop(columns=["ServiceDt", "ClaimBillDate", "f11insdob"])

    # -------- FREQUENCY ENCODING -------- #
    for col, mapping in freq_maps.items():
        df[col] = df[col].map(mapping).fillna(0)

    # -------- FINAL COLUMN ALIGNMENT -------- #
    df = df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

    return df

# ---------------- ROUTES ---------------- #

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = preprocess_input(data)

        # -------- Binary -------- #
        binary_pred = binary_model.predict(df)[0]

        # -------- Multi-class -------- #
        multi_pred = multiclass_model.predict(df)[0]
        multi_label = le.inverse_transform([multi_pred])[0]

        # -------- Multi-label -------- #
        df_ml = df.drop(columns=[
            "service_year",
            
            "billing_year"
            
            
        ]).values

        multi_label_pred = multilabel_model.predict(df_ml)
        reasons = mlb.inverse_transform(multi_label_pred.reshape(1, -1))[0]

        return {
            "denial_flag": int(binary_pred),
            "denial_type": multi_label,
            "denial_reasons": list(reasons)
        }

    except Exception as e:
        return {"error": str(e)}